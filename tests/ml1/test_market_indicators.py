"""Indicator timing tests; no model fitting or predictive performance metrics."""

from dataclasses import replace
from pathlib import Path
import hashlib
import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from src.models.ml1.core import DataValidationError
from src.models.ml1.indicator_validation import (
    build_feature_variant_folds, get_feature_variant_fold_xy, load_feature_variant,
)
from src.models.ml1.market_indicators import (
    ENRICHED_COLUMNS, VARIANT_FILES, VIX_COLUMNS, build_vix_features,
    build_vix_variant, read_vix, variant_profile,
)
from src.models.ml1.pipeline import build_dataset
from src.models.ml1.time_validation import CONTROL_COLUMNS, FEATURE_COLUMNS


def history():
    sessions = xcals.get_calendar("ARCX", start="2014-12-01", end="2015-03-31").sessions
    vix = pd.Series(np.arange(len(sessions), dtype=float)+15, index=sessions, name="VIXCLS")
    return vix, sessions


def test_default_is_core_and_experimental_requires_explicit_opt_in():
    assert variant_profile()["name"] == "core"
    assert variant_profile()["default"] is True
    with pytest.raises(DataValidationError, match="experimental"):
        variant_profile("core_vix")
    profile = variant_profile("core_vix", True)
    assert profile["status"] == "experimental"
    assert profile["point_in_time_certified"] is False


def test_macro_variant_fails_closed_even_with_experimental_opt_in():
    with pytest.raises(DataValidationError, match="Macro variant blocked"):
        variant_profile("core_vix_macro", True)


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_lag_and_five_session_change_use_spy_session_positions(lag):
    vix, sessions = history()
    result = build_vix_features(vix, sessions, lag)
    expected_level = vix.shift(lag).rename("VIX_level")
    expected_level.index = expected_level.index.rename("Date")
    pd.testing.assert_series_equal(result.features.VIX_level, expected_level)
    expected_change = (vix.shift(lag)/vix.shift(lag+5)-1).rename("VIX_change_5d")
    expected_change.index = expected_change.index.rename("Date")
    pd.testing.assert_series_equal(result.features.VIX_change_5d, expected_change)
    assert result.audit["point_in_time_certified"] is False


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_same_day_or_invalid_lag_is_not_allowed_for_unverified_snapshot(bad):
    vix, sessions = history()
    with pytest.raises(ValueError, match="at least one SPY session"):
        build_vix_features(vix, sessions, bad)


def test_shift_happens_after_spy_reindex_not_on_fred_business_days():
    vix, sessions = history()
    # NYSE is closed on MLK Monday; a provider-only value must be ignored.
    extra = pd.Series([9999.], index=pd.DatetimeIndex(["2015-01-19"]), name="VIXCLS")
    source = pd.concat([vix, extra]).sort_index()
    result = build_vix_features(source, sessions)
    assert result.features.loc["2015-01-20", "VIX_level"] == vix.loc["2015-01-16"]
    assert result.audit["valid_non_spy_observations_ignored"] == 1
    assert pd.Timestamp("2015-01-19") not in result.features.index


def test_no_backfill_no_calendar_deletion_and_no_implicit_forward_fill():
    vix, sessions = history()
    vix = vix.drop(sessions[10])
    result = build_vix_features(vix, sessions)
    assert result.features.index.equals(sessions)
    assert pd.isna(result.features.loc[sessions[0], "VIX_level"])
    assert pd.isna(result.features.loc[sessions[11], "VIX_level"])
    assert pd.isna(result.features.loc[sessions[11], "VIX_change_5d"])
    assert pd.isna(result.features.loc[sessions[16], "VIX_change_5d"])
    assert result.features.loc[sessions[12], "VIX_level"] == vix.loc[sessions[11]]
    assert result.audit["missing_on_spy_sessions_before_lag"] == 1


@pytest.mark.parametrize("invalid", [0, -1, np.inf, -np.inf, np.nan])
def test_invalid_vix_does_not_become_a_zero_or_fabricated_level(invalid):
    vix, sessions = history()
    vix.iloc[10] = invalid
    result = build_vix_features(vix, sessions)
    assert pd.isna(result.features.VIX_level.iloc[11])


def test_publication_after_decision_is_rejected_despite_one_session_lag():
    vix, sessions = history()
    publication = pd.Series(sessions.tz_localize("America/New_York")+pd.Timedelta(hours=16), index=sessions)
    # An observation dated before the decision can still have a late publication.
    publication.iloc[10] = (sessions[11]+pd.Timedelta(hours=18)).tz_localize("America/New_York")
    result = build_vix_features(vix, sessions, known_at=publication)
    assert pd.isna(result.features.VIX_level.iloc[11])
    assert result.lineage.publication_unavailable_or_future.iloc[11]
    assert result.features.VIX_level.iloc[12] == vix.iloc[11]
    assert result.audit["point_in_time_certified"] is False


def test_missing_publication_evidence_does_not_fall_back_to_an_assumed_time():
    vix, sessions = history()
    publication = pd.Series(pd.NaT, index=sessions, dtype="datetime64[ns, UTC]")
    result = build_vix_features(vix, sessions, known_at=publication)
    assert result.features.isna().all().all()
    assert result.audit["known_at_evidence_supplied"] is True


def test_naive_publication_timestamp_is_rejected():
    vix, sessions = history()
    publication = pd.Series(sessions+pd.Timedelta(hours=16), index=sessions)
    with pytest.raises(DataValidationError, match="explicit timezone"):
        build_vix_features(vix, sessions, known_at=publication)


@pytest.mark.parametrize("cut", [10, 30, 60])
def test_truncating_future_does_not_change_historical_indicator_features(cut):
    vix, sessions = history()
    full = build_vix_features(vix, sessions)
    truncated = build_vix_features(vix.iloc[:cut+1], sessions[:cut+1])
    pd.testing.assert_frame_equal(full.features.iloc[:cut+1], truncated.features)


def test_inputs_are_not_mutated_and_unknown_known_at_is_not_invented():
    vix, sessions = history()
    before = vix.copy()
    index_before = sessions.copy()
    result = build_vix_features(vix, sessions)
    pd.testing.assert_series_equal(vix, before)
    pd.testing.assert_index_equal(sessions, index_before)
    assert result.lineage.VIX_known_at.isna().all()
    assert not result.lineage.publication_evidence_supplied.any()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("indicator-variants")
    raw, base, enriched = root/"raw", root/"core", root/"vix"
    raw.mkdir()
    sessions = xcals.get_calendar("ARCX", start="2004-01-01", end="2024-12-31").sessions
    n = np.arange(len(sessions))
    for ticker, values in [("SPY", 100*np.exp(.0001*n+.02*np.sin(n/7))),
                           ("IEF", 80*np.exp(.00005*n+.01*np.cos(n/11)))]:
        pd.DataFrame({"Date":sessions.strftime("%Y-%m-%d"),"Adj Close":values}).to_csv(raw/f"{ticker}_raw.csv",index=False)
    pd.DataFrame({"observation_date":sessions.strftime("%Y-%m-%d"),"VIXCLS":15+np.sin(n/23)}).to_csv(raw/"VIX_raw.csv",index=False)
    build_dataset(raw, base)
    before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
    metadata = build_vix_variant(base, raw, enriched, allow_experimental=True)
    return raw, base, enriched, before, metadata


def test_variant_export_preserves_core_files_dates_and_targets(workspace):
    raw, base, enriched, before, metadata = workspace
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()} == before
    assert set(p.name for p in enriched.iterdir()) == set(VARIANT_FILES)
    X = pd.read_parquet(enriched/"features_ml1.parquet")
    original = pd.read_parquet(base/"features_ml1.parquet")
    pd.testing.assert_frame_equal(X.loc[:,list(FEATURE_COLUMNS)],original)
    assert X.index.equals(original.index)
    assert tuple(X.columns) == ENRICHED_COLUMNS
    for name in ["targets_ml1.parquet","target_diagnostics.csv"]:
        assert (enriched/name).read_bytes() == (base/name).read_bytes()
    assert metadata["profile"]["status"] == "experimental"
    assert metadata["vix_source"]["point_in_time_certified"] is False
    assert metadata["default_variant"] == "core"


def test_default_adapter_is_identical_to_core_walk_forward(workspace):
    raw, base, _, _, _ = workspace
    data = load_feature_variant(base_dir=base, raw_dir=raw)
    assert tuple(data.X.columns) == FEATURE_COLUMNS
    assert len(build_feature_variant_folds(data)) == 8


def test_enriched_adapter_has_eight_safe_folds_and_no_future_metadata(workspace):
    raw, base, enriched, _, _ = workspace
    data = load_feature_variant("core_vix",allow_experimental=True,base_dir=base,variant_dir=enriched,raw_dir=raw)
    assert data.core.y.index.max() < pd.Timestamp("2023-01-01")
    assert len(data.core.boundary_excluded_dates) == 21
    folds = build_feature_variant_folds(data)
    assert len(folds) == 8
    for fold in folds:
        X_train,y_train,X_validation,y_validation = get_feature_variant_fold_xy(data,fold)
        assert tuple(X_train.columns) == ENRICHED_COLUMNS == tuple(X_validation.columns)
        assert not set(CONTROL_COLUMNS)&set(X_train.columns)
        assert y_train.name == y_validation.name == "y_crisis_21d"
        assert (data.core.target_info.iloc[fold.train_indices].target_end_session < X_validation.index[0]).all()


def test_variant_loader_never_reads_final_test_labels(workspace,monkeypatch):
    raw, base, enriched, _, _ = workspace
    reader = pd.read_parquet

    def guarded(path,*args,**kwargs):
        if Path(path).name == "targets_ml1.parquet":
            columns = kwargs.get("columns")
            assert columns is not None
            if "y_crisis_21d" in columns:
                assert Path(path).parent == base
                assert ("Date","<",pd.Timestamp("2023-01-01")) in kwargs.get("filters",[])
                assert ("target_end_session","<",pd.Timestamp("2023-01-01")) in kwargs.get("filters",[])
        return reader(path,*args,**kwargs)

    monkeypatch.setattr(pd,"read_parquet",guarded)
    load_feature_variant("core_vix",allow_experimental=True,base_dir=base,variant_dir=enriched,raw_dir=raw)


def test_experimental_variant_cannot_be_silently_promoted(workspace):
    raw, base, enriched, _, _ = workspace
    data = load_feature_variant("core_vix",allow_experimental=True,base_dir=base,variant_dir=enriched,raw_dir=raw)
    promoted = replace(data,profile={**data.profile,"status":"reference"})
    with pytest.raises(DataValidationError,match="must not be represented as operational"):
        build_feature_variant_folds(promoted)


def test_variant_build_never_overwrites_existing_outputs(workspace):
    raw, base, enriched, _, _ = workspace
    with pytest.raises(FileExistsError):
        build_vix_variant(base,raw,enriched,allow_experimental=True)


def test_variant_build_rejects_core_output_directory(workspace):
    raw, base, _, _, _ = workspace
    with pytest.raises(DataValidationError,match="separate"):
        build_vix_variant(base,raw,base,allow_experimental=True)


def test_build_keeps_missing_vix_rows_and_never_connects_to_network(workspace,tmp_path,monkeypatch):
    import socket
    raw, base, _, _, _ = workspace
    changed_raw = tmp_path/"raw"
    changed_raw.mkdir()
    # Preserve SPY provenance, but create a separate synthetic indicator scenario.
    for name in ["SPY_raw.csv","VIX_raw.csv"]:
        (changed_raw/name).write_bytes((raw/name).read_bytes())
    source = pd.read_csv(changed_raw/"VIX_raw.csv")
    source.loc[500,"VIXCLS"] = np.nan
    source.to_csv(changed_raw/"VIX_raw.csv",index=False)

    def no_network(*args,**kwargs):
        raise AssertionError("Indicator build must be offline")

    monkeypatch.setattr(socket.socket,"connect",no_network)
    out = tmp_path/"out"
    build_vix_variant(base,changed_raw,out,allow_experimental=True)
    X = pd.read_parquet(out/"features_ml1.parquet")
    assert X.index.equals(pd.read_parquet(base/"features_ml1.parquet").index)
    assert X.VIX_level.isna().sum() == 1
    assert X.VIX_change_5d.isna().sum() == 2
    data = load_feature_variant("core_vix",allow_experimental=True,base_dir=base,variant_dir=out,raw_dir=changed_raw)
    assert len(data.excluded_incomplete_feature_dates) == 2
    assert np.isfinite(data.X).all().all()


def test_variant_exports_are_reproducible(workspace,tmp_path):
    raw, base, enriched, _, _ = workspace
    out = tmp_path/"second"
    build_vix_variant(base,raw,out,allow_experimental=True)
    for name in VARIANT_FILES:
        assert (enriched/name).read_bytes() == (out/name).read_bytes(),name
