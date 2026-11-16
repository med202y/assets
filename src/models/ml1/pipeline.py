"""Offline CLI: python -m src.models.ml1.pipeline. Never trains a model."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from . import __version__
from .core import (
    DECISION_HOUR, DECISION_TIMEZONE, EXCLUDED_FEATURES, HORIZON,
    DataValidationError, build_features, build_targets, decision_times,
    target_diagnostics, validate_index, validate_spy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = PROJECT_ROOT / "src/data/financial_dataset/raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/ml1"
OUTPUT_FILES = (
    "features_ml1.parquet", "targets_ml1.parquet", "features_ml1_inference.parquet",
    "target_diagnostics.csv", "dataset_metadata.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_dates(values: pd.Series) -> pd.DatetimeIndex:
    """Use New York calendar dates for aware ETF timestamps, not UTC dates."""
    dates = []
    for value in values:
        try:
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("NaT")
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert(DECISION_TIMEZONE).tz_localize(None)
            elif stamp != stamp.normalize():
                raise ValueError("Naive intraday timestamps are ambiguous")
            dates.append(stamp.normalize())
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Invalid source date: {value!r}") from exc
    index = pd.DatetimeIndex(dates, name="Date")
    validate_index(index)
    return index


def read_prices(path: Path, ticker: str) -> pd.Series:
    data = pd.read_csv(path)
    if not {"Date", "Adj Close"}.issubset(data):
        raise DataValidationError(f"{path.name}: Date and Adj Close required")
    index = session_dates(data["Date"])
    values = pd.to_numeric(data["Adj Close"], errors="coerce").to_numpy(dtype=float)
    result = pd.Series(values, index=index, name=ticker)
    if ticker == "SPY":
        validate_spy(result)
    return result


def inspect_sources(raw_dir: Path) -> list[dict]:
    records = []
    for path in sorted(raw_dir.glob("*_raw.csv")):
        data = pd.read_csv(path)
        date_col = "Date" if "Date" in data else "observation_date"
        if date_col not in data:
            raise DataValidationError(f"Date column missing in {path.name}")
        index = session_dates(data[date_col])
        is_etf = "Adj Close" in data
        value_col = "Adj Close" if is_etf else data.columns[1]
        values = pd.Series(pd.to_numeric(data[value_col], errors="coerce").to_numpy(), index=index)
        ticker = path.name.removesuffix("_raw.csv")
        valid = values.where(np.isfinite(values))
        records.append({
            "file": path.name, "sha256": sha256(path), "columns": list(data.columns),
            "rows": len(data), "value_column": value_col,
            "first_date": str(index.min().date()), "last_date": str(index.max().date()),
            "valid_values": int(valid.notna().sum()), "missing_values": int(valid.isna().sum()),
            "first_valid_date": str(valid.first_valid_index().date()) if valid.notna().any() else None,
            "last_valid_date": str(valid.last_valid_index().date()) if valid.notna().any() else None,
            "unit": "adjusted USD" if is_etf else ("index points" if ticker == "VIX" else "percentage points"),
            "used_as_feature_input": ticker in {"SPY", "IEF"},
            "publication_evidence": "daily market-close convention" if is_etf else "not present",
        })
    return records


def check_calendar(index: pd.DatetimeIndex) -> dict:
    """ARCX aliases XNYS in this library; compare, do not fabricate missing prices."""
    validate_index(index)
    calendar = xcals.get_calendar("ARCX", start=index.min(), end=index.max())
    expected = calendar.sessions_in_range(index.min(), index.max()).tz_localize(None)
    missing = expected.difference(index)
    unexpected = index.difference(expected)
    if len(missing) or len(unexpected):
        raise DataValidationError(
            f"Unresolved SPY calendar anomaly: missing={missing.strftime('%Y-%m-%d').tolist()}, "
            f"unexpected={unexpected.strftime('%Y-%m-%d').tolist()}; no prices filled"
        )
    closed_weekdays = pd.bdate_range(index.min(), index.max()).difference(expected)
    closes = calendar.closes.reindex(expected)
    cutoffs = decision_times(index)
    if not (closes.to_numpy() < pd.DatetimeIndex(cutoffs).to_numpy()).all():
        raise DataValidationError("Decision cutoff must follow each scheduled market close")
    return {
        "requested_calendar": "ARCX", "resolved_calendar": calendar.name,
        "implementation": type(calendar).__name__, "library": "exchange-calendars",
        "library_version": xcals.__version__, "expected_sessions": len(expected),
        "observed_spy_sessions": len(index), "missing_sessions": [], "unexpected_sessions": [],
        "closed_weekdays": closed_weekdays.strftime("%Y-%m-%d").tolist(),
        "scheduled_close_before_cutoff": True,
        "limitation": "Library rules, including ARCX alias to XNYS; not an official exchange audit",
    }


def period(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "first_date": str(frame.index.min().date()),
            "last_date": str(frame.index.max().date()), "columns": list(frame.columns)}


def build_dataset(raw_dir: Path = DEFAULT_RAW, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    # Prevent arbitrary destinations inside shared source, notebook, test or report trees.
    if output_dir.is_relative_to(PROJECT_ROOT) and not output_dir.is_relative_to(DEFAULT_OUTPUT):
        raise DataValidationError("Repository outputs must stay within data/processed/ml1/")
    if output_dir == raw_dir or output_dir.is_relative_to(raw_dir) or raw_dir.is_relative_to(output_dir):
        raise DataValidationError("Output directory must not overlap raw inputs")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}; choose a new ML1 run directory")

    inventory = inspect_sources(raw_dir)
    spy = read_prices(raw_dir / "SPY_raw.csv", "SPY")
    calendar = check_calendar(spy.index)
    ief_path = raw_dir / "IEF_raw.csv"
    ief = read_prices(ief_path, "IEF") if ief_path.exists() else None
    features_all = build_features(spy, ief)
    targets_all = build_targets(spy)
    inference = features_all.loc[features_all.notna().all(axis=1)].copy()
    supervised_index = inference.index.intersection(targets_all.index[targets_all.y_crisis_21d.notna()])
    features = inference.loc[supervised_index].copy()
    # Candidate labels only feed training-period diagnostics, never operational feature/target files.
    targets = targets_all.loc[supervised_index].drop(columns=["y_crisis_21d_07", "y_crisis_21d_10"])
    diagnostics = target_diagnostics(targets_all)
    if features.empty or inference.empty:
        raise DataValidationError("Insufficient history or valid features; no tables generated")
    if not features.index.equals(targets.index) or set(features) & set(targets):
        raise DataValidationError("Feature/target alignment or column separation failed")
    if not (targets.future_session_count == HORIZON).all():
        raise DataValidationError("Incomplete labelled horizon")
    pd.testing.assert_frame_equal(features, inference.loc[features.index])

    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "pyarrow", "exchange-calendars"]}
    code_hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    limitations = [
        "7% is provisional by user instruction, not chosen using final-test outcomes.",
        "Macro and VIX excluded: historical publication/receipt times and vintages absent.",
        "ETF adjusted-close snapshots may be revised; historical receipt at 17:00 is assumed, not logged.",
        "Exchange-calendar library is a reference with ARCX resolved to XNYS, not an official audit.",
        "Daily adjusted closes do not detect intraday threshold crossings.",
        "Overlapping labels are not independent crisis episodes.",
        "Supervised exports include final-test dates: future training must use explicit time splits and purging.",
    ]
    metadata = {
        "pipeline_version": __version__, "python": platform.python_version(), "dependencies": versions,
        "pipeline_source_sha256": code_hashes, "sources": inventory, "calendar": calendar,
        "target": {"name": "y_crisis_21d", "operational_threshold": 0.07,
                   "threshold_status": "provisional_user_instruction", "diagnostic_thresholds": [0.07, 0.10],
                   "horizon_spy_sessions": HORIZON, "comparison": "strictly below (1-x)*P[t]",
                   "formula": "min(P[t+1:t+22])/P[t]-1 < -x", "incomplete_horizon": "NA, never zero",
                   "price_column": "Adj Close", "legacy_stress_label_used": False},
        "temporal_policy": {
            "decision_time": f"{DECISION_HOUR}:00", "timezone": DECISION_TIMEZONE,
            "price_availability": "assume receipt of each market close by cutoff; otherwise no decision",
            "early_close_policy": "same 17:00 local cutoff", "macro_policy": "exclude without historical evidence",
            "imputation": "none", "scaling_fitted": False, "statistical_selection_fitted": False,
            "diagnostic_period": ["2005-01-01", "2014-12-31"],
            "diagnostic_target_end_max": "2014-12-31", "final_test_start": "2023-01-01",
            "test_outcomes_used_for_selection": False,
            "training_rule": "2005-2014 decisions AND target_end_session <= 2014-12-31; purge later split boundaries",
        },
        "features_retained": list(features.columns), "features_excluded": EXCLUDED_FEATURES,
        "tables": {"features_ml1": period(features), "targets_ml1": period(targets),
                   "features_ml1_inference": period(inference)},
        "quality": {
            "all_spy_sessions_preserved": len(features_all) == len(spy),
            "raw_spy_rows": len(spy), "feature_warmup_or_unavailable_rows": len(spy) - len(inference),
            "inference_rows_without_known_target": int(targets_all.loc[inference.index, "y_crisis_21d"].isna().sum()),
            "latest_raw_date": str(spy.index[-1].date()), "latest_inference_date": str(inference.index[-1].date()),
            "ief_unavailable_on_spy_sessions": int(ief.reindex(spy.index).where(lambda s: np.isfinite(s) & (s > 0)).isna().sum()) if ief is not None else None,
            "labelled_future_counts": [int(v) for v in targets.future_session_count.unique()],
        },
        "limitations": limitations, "output_files": list(OUTPUT_FILES),
    }
    if ief is None:
        metadata["features_excluded"] = {**EXCLUDED_FEATURES, "SPY_IEF_corr_63d": "IEF raw file absent"}
        metadata["limitations"].append("IEF missing: only the nine SPY-only features retained.")

    # Stage a complete run and rename it; never overwrite any existing run.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        features.to_parquet(stage / "features_ml1.parquet", engine="pyarrow")
        targets.to_parquet(stage / "targets_ml1.parquet", engine="pyarrow")
        inference.to_parquet(stage / "features_ml1_inference.parquet", engine="pyarrow")
        diagnostics.to_csv(stage / "target_diagnostics.csv", index=False)
        metadata["output_sha256"] = {name: sha256(stage / name) for name in OUTPUT_FILES[:-1]}
        (stage / "dataset_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        # A concurrent run or user-created directory must not be replaced.
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during build: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="New run directory only; existing outputs are never overwritten")
    args = parser.parse_args()
    try:
        metadata = build_dataset(args.raw_dir, args.output_dir)
    except (DataValidationError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"ML1 validation failed: {exc}\n")
    print(json.dumps({"output_dir": str(args.output_dir), "tables": metadata["tables"],
                      "quality": metadata["quality"]}, indent=2))


if __name__ == "__main__":
    main()
