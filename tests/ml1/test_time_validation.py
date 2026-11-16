"""Topology tests use synthetic labels; real integration never reads 2023+ labels."""

from dataclasses import replace
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from src.models.ml1.core import DataValidationError
from src.models.ml1.time_validation import (
    CONTROL_COLUMNS, FEATURE_COLUMNS, FINAL_TEST_START, LABEL,
    InsufficientFoldError, WalkForwardConfig, apply_post_validation_embargo,
    assert_fold_is_safe, build_walk_forward_folds, get_fold_xy,
    load_development_data, prepare_development_data,
)


def synthetic_tables(sessions, sample_dates):
    positions = sessions.get_indexer(sample_dates)
    X = pd.DataFrame({c: np.arange(len(sample_dates), dtype=float)/100 for c in FEATURE_COLUMNS}, index=sample_dates)
    y = pd.Series(0, index=sample_dates, dtype="Int8", name=LABEL)
    controls = pd.DataFrame({
        "target_start_session": sessions[positions+1],
        "target_end_session": sessions[positions+21],
        "future_session_count": 21,
    }, index=sample_dates)
    return X, y, controls


@pytest.fixture
def tables():
    sessions = xcals.get_calendar("ARCX", start="2004-01-01", end="2024-12-31").sessions
    dates = sessions[(sessions >= "2005-01-01") & (sessions < "2024-01-01")]
    X, y, controls = synthetic_tables(sessions, dates)
    return X, y, controls, sessions


@pytest.fixture
def data(tables):
    return prepare_development_data(*tables)


def test_eight_annual_folds_strict_chronology_and_purge(data):
    folds = build_walk_forward_folds(data)
    assert [f.validation_year for f in folds] == list(range(2015, 2023))
    for fold in folds:
        train_dates = data.X.index[fold.train_indices]
        validation_dates = data.X.index[fold.validation_indices]
        assert train_dates.is_monotonic_increasing and validation_dates.is_monotonic_increasing
        assert train_dates[-1] < validation_dates[0]
        assert train_dates.intersection(validation_dates).empty
        assert len(fold.purged_train_indices) == 21
        assert (data.target_info.iloc[fold.train_indices].target_end_session < validation_dates[0]).all()
        assert fold.latest_train_label_known_at < fold.first_validation_decision
        assert_fold_is_safe(data, fold)
    for previous, current in zip(folds, folds[1:]):
        assert set(previous.train_indices) < set(current.train_indices)


def test_2014_2015_boundary_is_target_end_not_just_decision_date(data):
    first = build_walk_forward_folds(data)[0]
    assert first.validation_start == pd.Timestamp("2015-01-02")
    assert first.train_end == pd.Timestamp("2014-12-01")
    assert first.train_target_end_max == pd.Timestamp("2014-12-31")
    rejected = data.target_info.iloc[first.purged_train_indices]
    assert rejected.index.min() == pd.Timestamp("2014-12-02")
    assert rejected.target_end_session.min() == first.validation_start


def test_exactly_21_spy_sessions_are_purged_not_21_calendar_days(data):
    fold = build_walk_forward_folds(data)[0]
    first_validation_pos = data.spy_sessions.get_loc(fold.validation_start)
    last_training_pos = data.spy_sessions.get_loc(fold.train_end)
    assert first_validation_pos-last_training_pos == 22
    # Weekends and holidays make this interval longer than 22 calendar days.
    assert (fold.validation_start-fold.train_end).days > 22


def test_target_reaching_first_validation_session_cannot_be_reintroduced(data):
    fold = build_walk_forward_folds(data)[0]
    unsafe = np.sort(np.append(fold.train_indices, fold.purged_train_indices[0]))
    with pytest.raises(DataValidationError, match="reaches validation"):
        assert_fold_is_safe(data, replace(fold, train_indices=unsafe))


def test_calendar_check_is_independent_of_target_end_comparison(data):
    fold = build_walk_forward_folds(data)[0]
    bad = fold.purged_train_indices[0]
    controls = data.target_info.copy()
    controls.iloc[bad, controls.columns.get_loc("target_end_session")] = pd.Timestamp("2014-12-31")
    corrupt = replace(data, target_info=controls)
    unsafe = np.sort(np.append(fold.train_indices, bad))
    with pytest.raises(DataValidationError, match="21-session purge"):
        assert_fold_is_safe(corrupt, replace(fold, train_indices=unsafe))


@pytest.mark.parametrize("embargo", [0, 7, 21])
def test_no_second_gap_in_expanding_window(data, embargo):
    folds = build_walk_forward_folds(data, WalkForwardConfig(embargo_sessions=embargo))
    baseline = build_walk_forward_folds(data)
    for fold, reference in zip(folds, baseline):
        np.testing.assert_array_equal(fold.train_indices, reference.train_indices)
        assert fold.embargo["applied"] is False
        assert fold.embargo["configured_sessions"] == embargo
        assert fold.embargo["excluded_train_observations"] == 0


def test_previous_validation_rows_can_enter_next_fold_when_labels_known(data):
    previous, current = build_walk_forward_folds(data)[:2]
    previous_eligible = previous.validation_indices[
        (data.target_info.iloc[previous.validation_indices].target_end_session < current.validation_start).to_numpy()
    ]
    assert set(previous_eligible).issubset(current.train_indices)
    unavailable = previous.validation_indices[
        (data.target_info.iloc[previous.validation_indices].target_end_session >= current.validation_start).to_numpy()
    ]
    assert len(unavailable) == 21
    assert not set(unavailable) & set(current.train_indices)


@pytest.mark.parametrize("embargo", [0, 1, 7, 21])
def test_post_validation_embargo_for_mixed_sided_training(embargo):
    sessions = pd.bdate_range("2014-11-03", periods=100)
    dates = sessions[:80]
    validation = np.arange(20, 30)
    train = np.r_[np.arange(0, 20), np.arange(30, 80)]
    result = apply_post_validation_embargo(train, validation, dates, sessions, embargo)
    np.testing.assert_array_equal(result.excluded_indices, np.arange(30, 30+embargo))
    assert not set(result.retained_indices) & set(result.excluded_indices)
    assert len(result.retained_indices)+len(result.excluded_indices) == len(train)
    if embargo:
        assert result.start_session == sessions[30]
        assert result.end_session == sessions[29+embargo]


def test_embargo_uses_calendar_sessions_even_when_sample_rows_are_missing():
    sessions = pd.bdate_range("2015-01-02", periods=100)
    dates = sessions.delete([12, 14, 16, 18])
    validation = np.arange(0, 10)
    train = np.arange(10, len(dates))
    result = apply_post_validation_embargo(train, validation, dates, sessions, 21)
    assert result.end_session == sessions[30]
    assert len(result.excluded_indices) == 17
    assert dates[result.retained_indices[0]] == sessions[31]


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_embargo_rejected(bad):
    with pytest.raises(ValueError, match="nonnegative integer"):
        WalkForwardConfig(embargo_sessions=bad)


@pytest.mark.parametrize("years", [(2023,), (2015, 2023), (2016, 2015), (2015, 2015), ()])
def test_final_test_year_cannot_be_configured_as_validation(years):
    with pytest.raises(ValueError, match="2015–2022"):
        WalkForwardConfig(validation_years=years)


def test_final_test_and_cross_boundary_2022_labels_are_never_examined(tables):
    X, y, controls, sessions = tables

    class ForbiddenLabel:
        def __eq__(self, other):
            raise AssertionError("Final-test or cross-boundary outcome was inspected")

        def __float__(self):
            raise AssertionError("Final-test outcome was converted")

    poisoned = y.astype(object)
    excluded = (X.index >= FINAL_TEST_START) | (controls.target_end_session >= FINAL_TEST_START).to_numpy()
    for date in X.index[excluded]:
        poisoned.loc[date] = ForbiddenLabel()
    protected = prepare_development_data(X, poisoned, controls, sessions)
    folds = build_walk_forward_folds(protected)
    assert protected.X.index.max() == pd.Timestamp("2022-11-30")
    assert len(protected.boundary_excluded_dates) == 21
    assert protected.y.index.max() < FINAL_TEST_START
    assert folds[-1].validation_end == pd.Timestamp("2022-11-30")


def test_changing_final_labels_does_not_change_folds(tables):
    X, y, controls, sessions = tables
    first = prepare_development_data(X, y, controls, sessions)
    changed = y.copy()
    changed.loc[changed.index >= FINAL_TEST_START] = 1
    second = prepare_development_data(X, changed, controls, sessions)
    assert [f.summary() for f in build_walk_forward_folds(first)] == [f.summary() for f in build_walk_forward_folds(second)]


def test_repeated_folds_are_deterministic_and_indices_readonly(data):
    first, second = build_walk_forward_folds(data), build_walk_forward_folds(data)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.train_indices, b.train_indices)
        np.testing.assert_array_equal(a.validation_indices, b.validation_indices)
        assert a.summary() == b.summary()
        with pytest.raises(ValueError):
            a.train_indices[0] = 99


def test_small_synthetic_dataset():
    sessions = xcals.get_calendar("ARCX", start="2014-10-01", end="2015-04-30").sessions
    X, y, controls = synthetic_tables(sessions, sessions[:90])
    data = prepare_development_data(X, y, controls, sessions)
    config = WalkForwardConfig(validation_years=(2015,), min_train_observations=1, min_validation_observations=1)
    fold, = build_walk_forward_folds(data, config)
    assert len(fold.train_indices) > 0 and len(fold.validation_indices) > 0
    assert len(fold.purged_train_indices) == 21


@pytest.mark.parametrize("minimum,part", [("min_train_observations", "training"), ("min_validation_observations", "validation")])
def test_insufficient_fold_fails_explicitly(data, minimum, part):
    config = WalkForwardConfig(**{minimum: 100_000})
    with pytest.raises(InsufficientFoldError, match=part):
        build_walk_forward_folds(data, config)


def test_empty_annual_validation_is_not_silently_skipped(tables):
    X, y, controls, sessions = tables
    keep = X.index.year != 2018
    data = prepare_development_data(X.loc[keep], y.loc[keep], controls.loc[keep], sessions)
    with pytest.raises(InsufficientFoldError, match="Fold 2018.*validation"):
        build_walk_forward_folds(data)


def test_model_interface_returns_only_features_and_one_label(data):
    for fold in build_walk_forward_folds(data):
        X_train, y_train, X_validation, y_validation = get_fold_xy(data, fold)
        assert tuple(X_train.columns) == FEATURE_COLUMNS == tuple(X_validation.columns)
        assert y_train.name == LABEL == y_validation.name
        assert isinstance(y_train, pd.Series) and isinstance(y_validation, pd.Series)
        assert not set(CONTROL_COLUMNS) & set(X_train.columns)
        assert X_train.index.equals(y_train.index)
        assert X_validation.index.equals(y_validation.index)


def test_future_metadata_cannot_be_smuggled_into_X(tables):
    X, y, controls, sessions = tables
    X = X.copy()
    X["future_min_return_21"] = 0
    with pytest.raises(DataValidationError, match="exactly the ten"):
        prepare_development_data(X, y, controls, sessions)


def test_all_target_columns_cannot_be_passed_as_y(tables):
    X, y, controls, sessions = tables
    with pytest.raises(DataValidationError, match="single label Series"):
        prepare_development_data(X, y.to_frame().join(controls), controls, sessions)


def test_misaligned_feature_target_dates_fail(tables):
    X, y, controls, sessions = tables
    with pytest.raises(DataValidationError, match="not identical"):
        prepare_development_data(X.iloc[1:], y, controls, sessions)


def test_bad_target_calendar_metadata_fails(tables):
    X, y, controls, sessions = tables
    controls = controls.copy()
    end_position = sessions.get_loc(controls.iloc[0].target_end_session)
    controls.iloc[0, controls.columns.get_loc("target_end_session")] = sessions[end_position-1]
    with pytest.raises(DataValidationError, match="exactly t\\+1 through t\\+21"):
        prepare_development_data(X, y, controls, sessions)


def test_fold_indices_cannot_be_reused_on_another_dataset(data):
    fold = build_walk_forward_folds(data)[0]
    shifted = replace(data, X=data.X.iloc[1:], y=data.y.iloc[1:], target_info=data.target_info.iloc[1:])
    with pytest.raises(DataValidationError, match="different development dataset"):
        get_fold_xy(shifted, fold)


def test_future_metadata_added_after_fold_construction_still_cannot_reach_model(data):
    fold = build_walk_forward_folds(data)[0]
    X = data.X.copy()
    X["future_min_return_21"] = 123
    contaminated = replace(data, X=X)
    with pytest.raises(DataValidationError, match="separate features"):
        get_fold_xy(contaminated, fold)


def test_model_arrays_are_copies_not_mutable_views_of_development(data):
    fold = build_walk_forward_folds(data)[0]
    X_train, y_train, _, _ = get_fold_xy(data, fold)
    before = data.X.iloc[0, 0]
    X_train.iloc[0, 0] = -999
    y_train.iloc[0] = 1
    assert data.X.iloc[0, 0] == before
    assert data.y.iloc[0] == 0


def test_empty_mixed_training_can_be_embargoed_without_inventing_rows():
    dates = pd.bdate_range("2015-01-02", periods=80)
    embargo = apply_post_validation_embargo([], [0, 1], dates, dates)
    assert len(embargo.retained_indices) == len(embargo.excluded_indices) == 0


def test_temporal_controls_require_unique_increasing_end_dates(tables):
    X, y, controls, sessions = tables
    controls = controls.copy()
    controls.iloc[1, controls.columns.get_loc("target_end_session")] = controls.iloc[0].target_end_session
    with pytest.raises(DataValidationError, match="Invalid temporal control"):
        prepare_development_data(X, y, controls, sessions)


def test_real_loader_uses_filtered_label_reads_and_keeps_final_holdout_closed(monkeypatch):
    root = Path(__file__).resolve().parents[2] / "data/processed/ml1"
    if not (root/"dataset_metadata.json").exists():
        pytest.skip("Local generated ML1 dataset is not distributed in Git")
    original = pd.read_parquet
    reads = []

    def guarded_read(path, *args, **kwargs):
        columns = kwargs.get("columns")
        if Path(path).name == "targets_ml1.parquet":
            assert columns is not None, "Unfiltered read of all target columns is forbidden"
            if LABEL in columns:
                assert columns == [LABEL]
                filters = kwargs.get("filters", [])
                assert ("Date", "<", FINAL_TEST_START) in filters
                assert ("target_end_session", "<", FINAL_TEST_START) in filters
                reads.append(filters)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    data = load_development_data(root)
    folds = build_walk_forward_folds(data)
    assert len(reads) == 1
    assert len(folds) == 8
    assert len(data.boundary_excluded_dates) == 21
    assert data.y.index.max() == pd.Timestamp("2022-11-30")
    assert len(data.final_test_dates) > 0
