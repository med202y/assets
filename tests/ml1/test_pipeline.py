import hashlib
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from src.models.ml1.core import DataValidationError
from src.models.ml1.pipeline import (
    OUTPUT_FILES, build_dataset, check_calendar, read_prices, session_dates,
)


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    sessions = xcals.get_calendar("ARCX", start="2004-01-01", end="2006-12-31").sessions[:400]
    n = np.arange(len(sessions))
    for ticker, values in [
        ("SPY", 100*np.exp(.0003*n + .02*np.sin(n/5))),
        ("IEF", 80*np.exp(.0001*n + .01*np.cos(n/9))),
    ]:
        pd.DataFrame({"Date": sessions.strftime("%Y-%m-%d"), "Adj Close": values}).to_csv(root/f"{ticker}_raw.csv", index=False)
    return root


def test_pipeline_outputs_are_separate_aligned_and_latest_features_survive(raw, tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}
    out = tmp_path / "out"
    metadata = build_dataset(raw, out)
    assert set(p.name for p in out.iterdir()) == set(OUTPUT_FILES)
    x = pd.read_parquet(out / "features_ml1.parquet")
    y = pd.read_parquet(out / "targets_ml1.parquet")
    inference = pd.read_parquet(out / "features_ml1_inference.parquet")
    assert x.index.equals(y.index)
    assert not set(x) & set(y)
    assert not x.isna().any().any() and not y.y_crisis_21d.isna().any()
    assert np.isfinite(x).all().all()
    assert (y.future_session_count == 21).all()
    assert y.y_crisis_21d.dtype == "Int8"
    assert not {"y_crisis_21d_07", "y_crisis_21d_10", "y_stress_regime"} & set(y)
    pd.testing.assert_frame_equal(x, inference.loc[x.index])
    spy = read_prices(raw / "SPY_raw.csv", "SPY")
    assert inference.index[-1] == spy.index[-1]
    assert len(inference.index.difference(x.index)) == 21
    assert (len(spy), len(inference), len(x)) == (400, 149, 128)
    assert metadata["temporal_policy"]["test_outcomes_used_for_selection"] is False
    assert metadata["target"]["operational_threshold"] == .07
    assert metadata == json.loads((out/"dataset_metadata.json").read_text())
    for name, digest in metadata["output_sha256"].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()}


def test_existing_output_is_never_overwritten(raw, tmp_path):
    out = tmp_path / "out"
    build_dataset(raw, out)
    before = (out/"dataset_metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_dataset(raw, out)
    assert (out/"dataset_metadata.json").read_bytes() == before


def test_outputs_are_reproducible(raw, tmp_path):
    first, second = tmp_path/"first", tmp_path/"second"
    build_dataset(raw, first)
    build_dataset(raw, second)
    for name in OUTPUT_FILES:
        assert (first/name).read_bytes() == (second/name).read_bytes(), name


def test_macro_and_filtered_final_csv_never_become_feature_inputs(raw, tmp_path):
    first, second = tmp_path/"first", tmp_path/"second"
    build_dataset(raw, first)
    # Impossible/unavailable macro values and a false filtered CSV must not enter X/y.
    pd.DataFrame({"observation_date": ["2005-01-03"], "VIXCLS": [999999]}).to_csv(raw/"VIX_raw.csv", index=False)
    (raw/"final_dataset.csv").write_text("this must never be loaded")
    build_dataset(raw, second)
    for name in ["features_ml1.parquet", "targets_ml1.parquet", "features_ml1_inference.parquet"]:
        pd.testing.assert_frame_equal(pd.read_parquet(first/name), pd.read_parquet(second/name))
    metadata = json.loads((second/"dataset_metadata.json").read_text())
    assert "VIX" in metadata["features_excluded"]


def test_calendar_distinguishes_closures_and_missing_expected_session():
    sessions = xcals.get_calendar("ARCX", start="2012-10-25", end="2012-11-02").sessions
    info = check_calendar(sessions)
    assert "2012-10-29" in info["closed_weekdays"]
    assert "2012-10-30" in info["closed_weekdays"]
    assert info["missing_sessions"] == []
    with pytest.raises(DataValidationError, match="missing=.*2012-10-31"):
        check_calendar(sessions.drop(pd.Timestamp("2012-10-31")))


def test_calendar_rejects_unexpected_weekend():
    with pytest.raises(DataValidationError, match="unexpected=.*2012-11-03"):
        check_calendar(pd.DatetimeIndex(["2012-11-02", "2012-11-03", "2012-11-05"]))


def test_timezone_normalization_uses_new_york_session_date():
    result = session_dates(pd.Series(["2010-07-07T00:30:00+00:00"]))
    assert result[0] == pd.Timestamp("2010-07-06")


def test_ambiguous_naive_intraday_timestamp_rejected():
    with pytest.raises(DataValidationError, match="Invalid source date"):
        session_dates(pd.Series(["2010-07-07 15:00:00"]))


def test_missing_spy_session_fails_without_writing_output(raw, tmp_path):
    p = raw/"SPY_raw.csv"
    data = pd.read_csv(p).drop(index=100)
    data.to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="calendar anomaly"):
        build_dataset(raw, tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_ief_absence_uses_documented_spy_only_subset(raw, tmp_path):
    # Rename the fixture, not a shared input.
    (raw/"IEF_raw.csv").rename(raw/"IEF.unavailable")
    metadata = build_dataset(raw, tmp_path/"out")
    assert len(metadata["features_retained"]) == 9
    assert "SPY_IEF_corr_63d" in metadata["features_excluded"]


def test_invalid_spy_price_fails_without_output(raw, tmp_path):
    p = raw/"SPY_raw.csv"
    data = pd.read_csv(p)
    data.loc[10, "Adj Close"] = 0
    data.to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="Invalid SPY"):
        build_dataset(raw, tmp_path/"out")
    assert not (tmp_path/"out").exists()


def test_input_and_output_cannot_overlap(raw):
    with pytest.raises(DataValidationError, match="overlap"):
        build_dataset(raw, raw/"out")


def test_build_is_offline_and_never_connects_to_a_provider(raw, tmp_path, monkeypatch):
    import socket

    def forbidden_connection(*args, **kwargs):
        raise AssertionError("Network access is forbidden during ML1 preparation")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connection)
    build_dataset(raw, tmp_path/"offline")
