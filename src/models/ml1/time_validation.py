"""Chronological ML1 folds, target-aware purge and explicit session embargo.

No estimator, fitting, shuffling, performance metric or final-test label access.
Indices always address DevelopmentData.X/y, not the original Parquet row numbers.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .core import DataValidationError, HORIZON, decision_times, validate_index
from .pipeline import DEFAULT_OUTPUT, DEFAULT_RAW, check_calendar, session_dates, sha256

FINAL_TEST_START = pd.Timestamp("2023-01-01")
TRAIN_START = pd.Timestamp("2005-01-01")
INITIAL_TRAIN_END = pd.Timestamp("2014-12-31")
LABEL = "y_crisis_21d"
CONTROL_COLUMNS = ("target_start_session", "target_end_session", "future_session_count")
FEATURE_COLUMNS = (
    "SPY_ret_1d", "SPY_ret_5d", "SPY_ret_21d", "SPY_ret_63d",
    "SPY_vol_5d", "SPY_vol_21d", "SPY_vol_63d", "SPY_drawdown_252d",
    "SPY_vs_ma200", "SPY_IEF_corr_63d",
)


class InsufficientFoldError(DataValidationError):
    """A requested annual fold cannot meet its declared minimum sizes."""


@dataclass(frozen=True)
class WalkForwardConfig:
    validation_years: tuple[int, ...] = tuple(range(2015, 2023))
    embargo_sessions: int = 21
    min_train_observations: int = 252
    min_validation_observations: int = 100

    def __post_init__(self):
        years = self.validation_years
        if not years or tuple(sorted(set(years))) != years or any(
            type(year) is not int or not 2015 <= year <= 2022 for year in years
        ):
            raise ValueError("Validation years must be unique increasing integers within 2015–2022")
        _nonnegative_int(self.embargo_sessions, "embargo_sessions")
        for name in ("min_train_observations", "min_validation_observations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _readonly(values) -> np.ndarray:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _fingerprint(index: pd.DatetimeIndex) -> str:
    return hashlib.sha256(index.as_unit("ns").asi8.tobytes()).hexdigest()


def _positions(indices, sample_dates: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1:
        raise DataValidationError("Indices must be a one-dimensional integer array")
    if not len(values):
        return np.empty(0, dtype=np.int64)
    if values.dtype.kind not in "iu":
        raise DataValidationError("Indices must be a one-dimensional integer array")
    if len(values) and ((values < 0).any() or (values >= len(sample_dates)).any()
                        or (np.diff(values) <= 0).any()):
        raise DataValidationError("Indices must be unique, increasing and in bounds")
    return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class DevelopmentData:
    X: pd.DataFrame
    y: pd.Series
    target_info: pd.DataFrame
    spy_sessions: pd.DatetimeIndex
    final_test_dates: pd.DatetimeIndex
    boundary_excluded_dates: pd.DatetimeIndex
    contract: dict


@dataclass(frozen=True)
class EmbargoResult:
    retained_indices: np.ndarray
    excluded_indices: np.ndarray
    requested_sessions: int
    start_session: pd.Timestamp | None
    end_session: pd.Timestamp | None
    available_sessions: int


def apply_post_validation_embargo(
    train_indices, validation_indices, sample_dates: pd.DatetimeIndex,
    spy_sessions: pd.DatetimeIndex, embargo_sessions: int = 21,
) -> EmbargoResult:
    """For mixed-sided training: exclude the next N SPY sessions after validation.

    This is only the embargo component, not a complete purged cross-validator.
    The embargo starts after the last validation decision, never before it.
    Historical validation observations are not permanently banned in later folds.
    """
    _nonnegative_int(embargo_sessions, "embargo_sessions")
    validate_index(sample_dates)
    validate_index(spy_sessions)
    train = _positions(train_indices, sample_dates)
    validation = _positions(validation_indices, sample_dates)
    if not len(validation):
        raise DataValidationError("Embargo requires a nonempty validation window")
    if np.intersect1d(train, validation).size:
        raise DataValidationError("Train and validation dates overlap")
    if (spy_sessions.get_indexer(sample_dates) < 0).any():
        raise DataValidationError("Sample dates must belong to the SPY session calendar")
    last_position = spy_sessions.get_loc(sample_dates[validation[-1]])
    embargo_dates = spy_sessions[last_position + 1:last_position + 1 + embargo_sessions]
    excluded = sample_dates[train].isin(embargo_dates)
    return EmbargoResult(
        _readonly(train[~excluded]), _readonly(train[excluded]), embargo_sessions,
        embargo_dates[0] if len(embargo_dates) else None,
        embargo_dates[-1] if len(embargo_dates) else None, len(embargo_dates),
    )


def prepare_development_data(
    X: pd.DataFrame, y: pd.Series, target_info: pd.DataFrame,
    spy_sessions: pd.DatetimeIndex, contract: dict | None = None,
) -> DevelopmentData:
    """Keep only 2005–2022 rows whose entire target also ends before 2023.

    Final-test label values are never validated, converted, counted or inspected.
    y can already be filtered by the loader, or be a full Series in synthetic use.
    Future target metadata is kept separately and cannot enter get_fold_xy().
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise DataValidationError("X must be a feature DataFrame and y a single label Series")
    validate_index(X.index)
    validate_index(target_info.index)
    validate_index(y.index)
    validate_index(spy_sessions)
    if tuple(X.columns) != FEATURE_COLUMNS:
        raise DataValidationError("X must contain exactly the ten historical ML1 features in order")
    if y.name != LABEL:
        raise DataValidationError(f"y must be only the Series {LABEL}")
    if not X.index.equals(target_info.index):
        raise DataValidationError("Feature and target dates are not identical")
    if not set(CONTROL_COLUMNS).issubset(target_info):
        raise DataValidationError("Missing target session metadata")
    controls = target_info.loc[:, list(CONTROL_COLUMNS)].copy()
    for name in CONTROL_COLUMNS[:2]:
        try:
            temporal_index = pd.DatetimeIndex(controls[name])
            validate_index(temporal_index)
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f"Invalid temporal control: {name}") from exc
        controls[name] = temporal_index
    positions = spy_sessions.get_indexer(X.index)
    starts = spy_sessions.get_indexer(pd.DatetimeIndex(controls.target_start_session))
    ends = spy_sessions.get_indexer(pd.DatetimeIndex(controls.target_end_session))
    if ((positions < 0).any() or (starts != positions + 1).any()
            or (ends != positions + HORIZON).any()
            or not controls.future_session_count.eq(HORIZON).all()):
        raise DataValidationError("Targets must span exactly t+1 through t+21 SPY sessions")
    before_test = (X.index >= TRAIN_START) & (X.index < FINAL_TEST_START)
    eligible = before_test & (controls.target_end_session < FINAL_TEST_START).to_numpy()
    dates = X.index[eligible]
    if dates.empty:
        raise DataValidationError("No development observations before the final-test boundary")
    if not dates.isin(y.index).all():
        raise DataValidationError("Missing development labels")
    # Select dates BEFORE any operation on label values, including binary checks.
    development_y = y.loc[dates].copy()
    if development_y.isna().any() or not development_y.isin([0, 1]).all():
        raise DataValidationError("Development labels must be complete and binary")
    development_y = development_y.astype("Int8")
    development_X = X.loc[dates].copy()
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in development_X.dtypes):
        raise DataValidationError("Financial features must be numeric")
    if not np.isfinite(development_X.to_numpy(dtype=float)).all():
        raise DataValidationError("Development features must be finite")
    return DevelopmentData(
        development_X, development_y, controls.loc[dates].copy(), spy_sessions.copy(),
        X.index[X.index >= FINAL_TEST_START].copy(),
        X.index[before_test & ~eligible].copy(), dict(contract or {}),
    )


def load_development_data(
    dataset_dir: Path = DEFAULT_OUTPUT, spy_csv: Path = DEFAULT_RAW / "SPY_raw.csv",
) -> DevelopmentData:
    """Read label values with pre-2023 and target-end filters, never final labels."""
    dataset_dir, spy_csv = Path(dataset_dir), Path(spy_csv)
    metadata = json.loads((dataset_dir / "dataset_metadata.json").read_text())
    target = metadata.get("target", {})
    policy = metadata.get("temporal_policy", {})
    if (target.get("name") != LABEL or target.get("horizon_spy_sessions") != HORIZON
            or target.get("operational_threshold") != .07
            or target.get("legacy_stress_label_used") is not False
            or policy.get("decision_time") != "17:00"
            or policy.get("timezone") != "America/New_York"
            or policy.get("final_test_start") != "2023-01-01"
            or tuple(metadata.get("features_retained", [])) != FEATURE_COLUMNS):
        raise DataValidationError("Dataset contract differs from the ML1 project validation contract")
    for name in ("features_ml1.parquet", "targets_ml1.parquet"):
        expected = metadata.get("output_sha256", {}).get(name)
        if not expected or sha256(dataset_dir / name) != expected:
            raise DataValidationError(f"Dataset fingerprint mismatch: {name}")
    source = next((s for s in metadata["sources"] if s["file"] == "SPY_raw.csv"), None)
    if source is None or source["sha256"] != sha256(spy_csv):
        raise DataValidationError("SPY calendar source fingerprint mismatch")
    sessions = session_dates(pd.read_csv(spy_csv, usecols=["Date"])["Date"])
    check_calendar(sessions)
    schema = pq.ParquetFile(dataset_dir / "targets_ml1.parquet").schema_arrow
    if LABEL not in schema.names or not pa.types.is_integer(schema.field(LABEL).type):
        raise DataValidationError("Crisis label must have an integer Parquet schema")
    X = pd.read_parquet(dataset_dir / "features_ml1.parquet")
    controls = pd.read_parquet(dataset_dir / "targets_ml1.parquet", columns=list(CONTROL_COLUMNS))
    # Filter at the Parquet read interface; no unfiltered read of y is performed.
    labels = pd.read_parquet(
        dataset_dir / "targets_ml1.parquet", columns=[LABEL],
        filters=[("Date", ">=", TRAIN_START), ("Date", "<", FINAL_TEST_START),
                 ("target_end_session", "<", FINAL_TEST_START)],
    )[LABEL]
    return prepare_development_data(X, labels, controls, sessions, metadata)


@dataclass(frozen=True)
class WalkForwardFold:
    validation_year: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    purged_train_indices: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_target_end_max: pd.Timestamp
    latest_train_label_known_at: pd.Timestamp
    first_validation_decision: pd.Timestamp
    n_train_candidates: int
    embargo: dict
    index_sha256: str

    def summary(self) -> dict:
        return {
            "validation_year": self.validation_year,
            "train_start": str(self.train_start.date()), "train_end": str(self.train_end.date()),
            "train_target_end_max": str(self.train_target_end_max.date()),
            "n_train": len(self.train_indices), "n_purged": len(self.purged_train_indices),
            "validation_start": str(self.validation_start.date()),
            "validation_end": str(self.validation_end.date()), "n_validation": len(self.validation_indices),
            "embargo_applied": self.embargo["applied"], "n_embargoed": self.embargo["excluded_train_observations"],
        }


def assert_fold_is_safe(data: DevelopmentData, fold: WalkForwardFold) -> None:
    """Explicit checks also run under python -O; no dependence on assert removal."""
    if (tuple(data.X.columns) != FEATURE_COLUMNS or not isinstance(data.y, pd.Series)
            or data.y.name != LABEL or not data.X.index.equals(data.y.index)
            or not data.X.index.equals(data.target_info.index)):
        raise DataValidationError("Model interface must separate features, label and target metadata")
    if fold.index_sha256 != _fingerprint(data.X.index):
        raise DataValidationError("Fold indices address a different development dataset")
    train = _positions(fold.train_indices, data.X.index)
    validation = _positions(fold.validation_indices, data.X.index)
    if not len(train) or not len(validation):
        raise DataValidationError("A fold must have both train and validation observations")
    train_dates, validation_dates = data.X.index[train], data.X.index[validation]
    ends = data.target_info.iloc[train].target_end_session
    train_positions = data.spy_sessions.get_indexer(train_dates)
    validation_position = data.spy_sessions.get_loc(validation_dates[0])
    if np.intersect1d(train, validation).size or train_dates[-1] >= validation_dates[0]:
        raise DataValidationError("Train must be strictly earlier than validation with no common date")
    # Two independent checks: actual target dates AND canonical 21-session positions.
    if not (ends < validation_dates[0]).all():
        raise DataValidationError("A training target reaches validation")
    if not (train_positions + HORIZON < validation_position).all():
        raise DataValidationError("The 21-session purge is violated")
    known_at = decision_times(pd.DatetimeIndex(ends))
    first_decision = decision_times(validation_dates).iloc[0]
    if not (known_at < first_decision).all():
        raise DataValidationError("Training labels are not known before the first validation decision")
    if ((train_dates >= FINAL_TEST_START).any() or (validation_dates >= FINAL_TEST_START).any()
            or not (data.target_info.iloc[validation].target_end_session < FINAL_TEST_START).all()):
        raise DataValidationError("Final-test observations or outcomes entered development")
    if fold.validation_year == 2015 and not (ends <= INITIAL_TRAIN_END).all():
        raise DataValidationError("Initial training targets extend past 2014")


def build_walk_forward_folds(
    data: DevelopmentData, config: WalkForwardConfig | None = None,
) -> tuple[WalkForwardFold, ...]:
    """Eight annual expanding folds by default; no persistent post-fold embargo."""
    config = config or WalkForwardConfig()
    if not data.X.index.equals(data.y.index) or not data.X.index.equals(data.target_info.index):
        raise DataValidationError("Development X, y and temporal controls must be aligned")
    if tuple(data.X.columns) != FEATURE_COLUMNS or data.y.name != LABEL:
        raise DataValidationError("Features and target metadata are not separated")
    dates = data.X.index
    folds = []
    for year in config.validation_years:
        validation = np.flatnonzero(dates.year == year)
        if len(validation) < config.min_validation_observations:
            raise InsufficientFoldError(f"Fold {year}: insufficient validation ({len(validation)} observations)")
        first_date = dates[validation[0]]
        candidates = np.flatnonzero((dates >= TRAIN_START) & (dates < first_date))
        ends = data.target_info.iloc[candidates].target_end_session
        safe = (ends < first_date).to_numpy()
        train, purged = candidates[safe], candidates[~safe]
        if len(train) < config.min_train_observations:
            raise InsufficientFoldError(f"Fold {year}: insufficient training after purge ({len(train)} observations)")
        train_ends = data.target_info.iloc[train].target_end_session
        # Calling the generic helper demonstrates it removes zero rows in this topology.
        embargo = apply_post_validation_embargo(train, validation, dates, data.spy_sessions, config.embargo_sessions)
        if len(embargo.excluded_indices):
            raise DataValidationError("An expanding fold unexpectedly contains post-validation training")
        fold = WalkForwardFold(
            year, _readonly(train), _readonly(validation), _readonly(purged),
            dates[train[0]], dates[train[-1]], first_date, dates[validation[-1]],
            train_ends.max(), decision_times(pd.DatetimeIndex(train_ends)).max(),
            decision_times(dates[validation]).iloc[0], len(candidates),
            {"configured_sessions": config.embargo_sessions, "applied": False,
             "reason": "strict_expanding_window_has_no_post_validation_training",
             "excluded_train_observations": 0,
             "would_start": str(embargo.start_session.date()) if embargo.start_session is not None else None,
             "would_end": str(embargo.end_session.date()) if embargo.end_session is not None else None,
             "available_post_validation_sessions": embargo.available_sessions},
            _fingerprint(dates),
        )
        assert_fold_is_safe(data, fold)
        folds.append(fold)
    return tuple(folds)


def get_fold_xy(data: DevelopmentData, fold: WalkForwardFold):
    """Return X_train, y_train, X_validation, y_validation; never target controls."""
    assert_fold_is_safe(data, fold)
    return (
        data.X.iloc[fold.train_indices].copy(), data.y.iloc[fold.train_indices].copy(),
        data.X.iloc[fold.validation_indices].copy(), data.y.iloc[fold.validation_indices].copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spy-csv", type=Path, default=DEFAULT_RAW / "SPY_raw.csv")
    parser.add_argument("--embargo-sessions", type=int, default=21)
    args = parser.parse_args()
    data = load_development_data(args.dataset_dir, args.spy_csv)
    folds = build_walk_forward_folds(data, WalkForwardConfig(embargo_sessions=args.embargo_sessions))
    print(pd.DataFrame([fold.summary() for fold in folds]).to_string(index=False))
    print(f"Development: {len(data.X)} rows; final-test label values loaded: 0; "
          f"pre-2023 boundary rows excluded: {len(data.boundary_excluded_dates)}")


if __name__ == "__main__":
    main()
