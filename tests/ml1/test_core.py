import numpy as np
import pandas as pd
import pytest

from src.models.ml1.core import (
    DataValidationError, align_known_observations, build_features, build_targets,
    decision_times, target_diagnostics, validate_spy,
)


def prices(values):
    return pd.Series(values, index=pd.bdate_range("2010-01-04", periods=len(values)), dtype=float)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0, -1])
def test_spy_prices_are_never_imputed_or_silently_dropped(bad):
    spy = prices([100, bad, 101])
    with pytest.raises(DataValidationError, match="Invalid SPY"):
        validate_spy(spy)


@pytest.mark.parametrize("order", [[0, 0, 2], [1, 0, 2]])
def test_unique_increasing_dates_required(order):
    spy = prices([100, 101, 102]).iloc[order]
    with pytest.raises(DataValidationError, match="unique and strictly increasing"):
        build_targets(spy)


@pytest.mark.parametrize("base,ratio,suffix", [(100, .93, "07"), (3, .93, "07"), (100, .9, "10"), (3, .9, "10")])
def test_exact_equality_is_not_crisis(base, ratio, suffix):
    spy = prices([base] + [base * ratio] * 21)
    # Exact decimal values at these synthetic boundaries, without return subtraction.
    assert build_targets(spy).iloc[0][f"y_crisis_21d_{suffix}"] == 0


@pytest.mark.parametrize("low,suffix", [(92.999, "07"), (89.999, "10")])
def test_strict_breach_is_positive(low, suffix):
    assert build_targets(prices([100, low] + [100] * 20)).iloc[0][f"y_crisis_21d_{suffix}"] == 1


def test_window_excludes_t_and_t_plus_22():
    spy = prices([100] + [100] * 21 + [1])
    target = build_targets(spy).iloc[0]
    assert target.y_crisis_21d == 0
    assert target.future_min_price_21 == 100
    assert target.target_start_session == spy.index[1]
    assert target.target_end_session == spy.index[21]
    assert target.future_session_count == 21
    # Current price must not be part of the future minimum either.
    assert build_targets(prices([50] + [100] * 21)).iloc[0].future_min_price_21 == 100


@pytest.mark.parametrize("position", [1, 21])
def test_first_and_twenty_first_future_sessions_are_included(position):
    values = [100] * 22
    values[position] = 80
    assert build_targets(prices(values)).iloc[0].y_crisis_21d == 1


def test_rebound_does_not_erase_a_breach():
    target = build_targets(prices([100, 80] + [110] * 20)).iloc[0]
    assert target.y_crisis_21d == 1
    assert target.future_min_return_21 == pytest.approx(-.2)


def test_last_21_labels_are_missing_not_negative():
    targets = build_targets(prices(np.linspace(100, 110, 50)))
    assert targets.iloc[-21:].y_crisis_21d.isna().all()
    assert targets.iloc[:-21].y_crisis_21d.notna().all()


@pytest.mark.parametrize("position", [0, 1, 21])
@pytest.mark.parametrize("bad", [np.nan, 0, -1, np.inf])
def test_invalid_price_anywhere_in_required_horizon_invalidates_label(position, bad):
    values = [100] * 23
    values[position] = bad
    targets = build_targets(prices(values))
    assert pd.isna(targets.iloc[0].y_crisis_21d)
    assert pd.isna(targets.iloc[0].future_min_return_21)


def test_operational_label_is_07_and_10_is_subset():
    spy = prices([100, 92] + [100] * 21 + [85] + [100] * 25)
    targets = build_targets(spy)
    pd.testing.assert_series_equal(targets.y_crisis_21d, targets.y_crisis_21d_07, check_names=False)
    valid = targets.dropna(subset=["y_crisis_21d"])
    assert (valid.y_crisis_21d_10 <= valid.y_crisis_21d_07).all()


def histories():
    n = np.arange(420)
    spy = prices(100 * np.exp(.0003*n + .02*np.sin(n/5)))
    ief = prices(80 * np.exp(.0001*n + .01*np.cos(n/9)))
    return spy, ief


def test_feature_formulas_and_warmup():
    spy, ief = histories()
    features = build_features(spy, ief)
    date = spy.index[300]
    ret = spy.pct_change(fill_method=None)
    assert features.loc[date, "SPY_ret_21d"] == pytest.approx(spy.iloc[300]/spy.iloc[279]-1)
    assert features.loc[date, "SPY_vol_21d"] == pytest.approx(ret.iloc[280:301].std(ddof=1)*np.sqrt(252))
    assert features.loc[date, "SPY_drawdown_252d"] == pytest.approx(spy.iloc[300]/spy.iloc[49:301].max()-1)
    assert features.loc[date, "SPY_vs_ma200"] == pytest.approx(spy.iloc[300]/spy.iloc[101:301].mean()-1)
    assert features.dropna().index[0] == spy.index[251]


@pytest.mark.parametrize("cut", [260, 330, 419])
def test_features_do_not_change_when_future_is_removed(cut):
    spy, ief = histories()
    full = build_features(spy, ief)
    truncated = build_features(spy.iloc[:cut+1], ief.iloc[:cut+1])
    pd.testing.assert_frame_equal(full.iloc[:cut+1], truncated)


def test_missing_ief_retains_spy_calendar_and_does_not_fill_prices():
    spy, ief = histories()
    missing_date = spy.index[300]
    features = build_features(spy, ief.drop(missing_date))
    assert features.index.equals(spy.index)
    assert pd.isna(features.loc[missing_date, "SPY_IEF_corr_63d"])
    assert features.loc[missing_date, "SPY_ret_1d"] == pytest.approx(spy.iloc[300]/spy.iloc[299]-1)


def test_macro_gate_rejects_future_publications_and_later_revisions():
    dates = pd.DatetimeIndex(["2010-01-04", "2010-01-05"])
    decisions = decision_times(dates)
    observations = pd.DataFrame({
        "observation_date": pd.to_datetime(["2010-01-04"]*3),
        "known_at": ["2010-01-04T21:00:00Z", "2010-01-04T23:00:00Z", "2010-01-06T12:00:00Z"],
        "value": [1, 2, 999],
    })
    aligned = align_known_observations(observations, decisions)
    assert aligned.value.tolist() == [1, 2]
    assert (pd.to_datetime(aligned.known_at, utc=True) <= decisions).all()


def test_future_macro_observation_leaves_na_not_backfilled():
    dates = pd.DatetimeIndex(["2010-01-04"])
    source = pd.DataFrame({"observation_date": ["2010-01-04"],
                           "known_at": ["2010-01-05T15:00:00Z"], "value": [100]})
    assert align_known_observations(source, decision_times(dates)).value.isna().all()


def test_old_observation_revision_does_not_replace_newer_observation():
    dates = pd.DatetimeIndex(["2010-01-06"])
    source = pd.DataFrame({"observation_date": ["2010-01-05", "2010-01-04"],
                           "known_at": ["2010-01-05T21:00:00Z", "2010-01-06T21:00:00Z"],
                           "value": [5, 999]})
    assert align_known_observations(source, decision_times(dates)).value.iloc[0] == 5


@pytest.mark.parametrize("column", ["observation_date", "known_at", "value"])
def test_macro_requires_evidence_columns(column):
    source = pd.DataFrame({"observation_date": ["2010-01-04"], "known_at": ["2010-01-04T21:00:00Z"], "value": [1]}).drop(columns=column)
    with pytest.raises(DataValidationError, match="Publication evidence"):
        align_known_observations(source, decision_times(pd.DatetimeIndex(["2010-01-04"])))


def test_naive_publication_time_is_not_accepted_as_proof():
    source = pd.DataFrame({"observation_date": ["2010-01-04"], "known_at": ["2010-01-04 15:00"], "value": [1]})
    with pytest.raises(DataValidationError, match="explicit timezone"):
        align_known_observations(source, decision_times(pd.DatetimeIndex(["2010-01-04"])))


def test_cutoff_respects_dst_and_stays_17_new_york():
    times = decision_times(pd.DatetimeIndex(["2010-01-04", "2010-07-06"]))
    assert times.iloc[0].hour == 22
    assert times.iloc[1].hour == 21


def test_diagnostics_never_use_2015_or_final_test_outcomes():
    spy = pd.Series(100., index=pd.bdate_range("2014-01-01", "2024-12-31"))
    original = target_diagnostics(build_targets(spy))
    spy.loc["2015-01-01":] = 1
    poisoned = target_diagnostics(build_targets(spy))
    pd.testing.assert_frame_equal(original, poisoned)
    total = original.loc[original.period == "training_total"]
    assert total.positive_labels.sum() == 0
    assert total.last_target_end.max() <= "2014-12-31"
