"""Provisional raw LR integration: saved historical OOF versus blind frozen inference."""

import argparse
from dataclasses import asdict, dataclass
from datetime import date as Date, datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import exchange_calendars as xcals
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import validate_development_features, validate_development_labels
from .core import DataValidationError, build_features, decision_times, validate_index
from .evaluate_ml1 import _check_dates_and_probabilities, validate_training_provenance
from .pipeline import DEFAULT_OUTPUT, DEFAULT_RAW, PROJECT_ROOT, check_calendar, read_prices, sha256
from .time_validation import FEATURE_COLUMNS, FINAL_TEST_START, LABEL, build_walk_forward_folds, load_development_data
from .train_ml1 import build_estimator, configuration_id, configuration_spec, positive_probabilities

DEFAULT_RELEASE = PROJECT_ROOT/"data/processed/ml1_inference_release"
DEFAULT_TRAINING = PROJECT_ROOT/"data/processed/ml1_training"
RELEASE_FILES = ("logistic_regression_final.joblib", "model_manifest.json", "monthly_oof_probabilities.csv",
                 "monthly_blind_probabilities.csv", "integration_contract.json")
POLICY = {"model": "logistic_regression", "feature_variant": "core", "calibration": "none",
          "status": "provisional", "horizon": "21_spy_sessions"}
_DEFAULT_PREDICTOR = None


def parse_prediction_date(value):
    if not isinstance(value, (str, Date, datetime, pd.Timestamp)):
        raise DataValidationError("Prediction date must be an ISO date YYYY-MM-DD")
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise DataValidationError("Prediction date must be an ISO date YYYY-MM-DD")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Invalid prediction date") from exc
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise DataValidationError("Use a naive session date without intraday time or timezone")
    return stamp


def _portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT/path


def _fingerprint(table):
    return hashlib.sha256(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()).hexdigest()


def load_saved_oof(data, training_dir=DEFAULT_TRAINING):
    """Validate old fold/configuration provenance without any evaluation or fit."""
    training_dir = Path(training_dir)
    metadata_path = training_dir/"training_metadata.json"
    path = training_dir/"ml_models_oof_predictions.parquet"
    metadata = json.loads(metadata_path.read_text())
    if sha256(path) != metadata["prediction_sha256"]:
        raise DataValidationError("Saved OOF prediction fingerprint mismatch")
    for name, digest in metadata["source_code_sha256"].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise DataValidationError(f"Historical source code fingerprint mismatch: {name}")
    frame = pd.read_parquet(path)  # This file contains probabilities and provenance, no labels.
    _check_dates_and_probabilities(frame, "model", ("logistic_regression", "random_forest", "xgboost"))
    folds = build_walk_forward_folds(data)
    validate_training_provenance(frame, metadata, data, folds)
    result = frame.loc[frame.model.eq("logistic_regression")].sort_values("prediction_date").copy()
    expected = data.X.index[np.concatenate([f.validation_indices for f in folds])]
    if not pd.DatetimeIndex(result.prediction_date).equals(expected):
        raise DataValidationError("Historical OOF dates differ from existing chronological folds")
    return result.set_index("prediction_date"), metadata


class FeatureSource:
    """Label-free inference reader; verify sources and recompute only price prefixes."""
    def __init__(self, dataset_dir=DEFAULT_OUTPUT, raw_dir=DEFAULT_RAW):
        self.dataset_dir, self.raw_dir = Path(dataset_dir), Path(raw_dir)
        metadata_path = self.dataset_dir/"dataset_metadata.json"
        self.metadata = json.loads(metadata_path.read_text())
        if (tuple(self.metadata["features_retained"]) != FEATURE_COLUMNS
                or self.metadata["temporal_policy"]["decision_time"] != "17:00"
                or self.metadata["temporal_policy"]["timezone"] != "America/New_York"):
            raise DataValidationError("Inference feature contract differs from core policy")
        self.paths = [metadata_path, self.dataset_dir/"features_ml1_inference.parquet",
                      self.raw_dir/"SPY_raw.csv", self.raw_dir/"IEF_raw.csv"]
        expected = self.metadata["output_sha256"]["features_ml1_inference.parquet"]
        if sha256(self.paths[1]) != expected:
            raise DataValidationError("Inference feature fingerprint mismatch")
        for filename in ("SPY_raw.csv", "IEF_raw.csv"):
            source = next(s for s in self.metadata["sources"] if s["file"] == filename)
            if sha256(self.raw_dir/filename) != source["sha256"]:
                raise DataValidationError(f"Raw price fingerprint mismatch: {filename}")
        for name in ("core.py", "pipeline.py"):
            if sha256(Path(__file__).parent/name) != self.metadata["pipeline_source_sha256"][name]:
                raise DataValidationError(f"Feature implementation fingerprint mismatch: {name}")
        self.features = pd.read_parquet(self.paths[1])
        validate_index(self.features.index)
        if tuple(self.features.columns) != FEATURE_COLUMNS:
            raise DataValidationError("Inference requires exactly the ten ordered core features")
        if not np.isfinite(self.features.to_numpy(dtype=float)).all():
            raise DataValidationError("Inference features contain NaN or infinity")
        self.spy = read_prices(self.raw_dir/"SPY_raw.csv", "SPY")
        self.ief = read_prices(self.raw_dir/"IEF_raw.csv", "IEF")
        check_calendar(self.spy.index)
        if not self.features.index.isin(self.spy.index).all():
            raise DataValidationError("Feature dates must be observed SPY sessions")
        self.fingerprints = {_portable(p): sha256(p) for p in self.paths}

    def check_unchanged(self):
        for path, digest in self.fingerprints.items():
            if sha256(_resolve(path)) != digest:
                raise DataValidationError(f"Feature source changed since loading: {path}")

    def feature_at(self, date):
        stamp = parse_prediction_date(date)
        if stamp not in self.spy.index:
            raise DataValidationError("Requested date is not a locally observed SPY session; no substitution")
        if stamp not in self.features.index:
            raise DataValidationError("Core features unavailable for the exact requested session")
        self.check_unchanged()
        # Later price values are never passed to the feature transformation.
        prefix_spy = self.spy.loc[self.spy.index <= stamp]
        prefix_ief = self.ief.loc[self.ief.index <= stamp]
        causal = build_features(prefix_spy, prefix_ief).loc[[stamp]]
        saved = self.features.loc[[stamp]]
        if (tuple(causal.columns) != FEATURE_COLUMNS or tuple(saved.columns) != FEATURE_COLUMNS
                or not np.isfinite(causal.to_numpy()).all()
                or not np.allclose(causal.to_numpy(), saved.to_numpy(), rtol=1e-10, atol=1e-12)):
            raise DataValidationError("Saved features disagree with causal price-prefix reconstruction")
        return causal


def fit_frozen_model(data, *, expected_rows=4329):
    """One fixed raw LR fit, exclusively on admissible pre-2023 development."""
    validate_development_features(data.X, FEATURE_COLUMNS)
    validate_development_labels(data.y)
    if expected_rows is not None and len(data.X) != expected_rows:
        raise DataValidationError("Development row count changed; review the integration contract")
    if (tuple(data.X.columns) != FEATURE_COLUMNS or not data.X.index.equals(data.y.index)
            or data.X.index.min() < pd.Timestamp("2005-01-01")
            or data.X.index.max() >= FINAL_TEST_START
            or not data.target_info.target_end_session.lt(FINAL_TEST_START).all()
            or data.y.name != LABEL or set(data.y.unique()) != {0, 1}
            or not np.isfinite(data.X.to_numpy()).all()):
        raise DataValidationError("Frozen model requires finite core development and mature pre-2023 labels")
    model = build_estimator("logistic_regression")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data.X, data.y.to_numpy(dtype=np.int8))
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise DataValidationError("Frozen LR did not converge; no parameter retuning permitted")
    return model, [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]


def integration_contract():
    return {"schema_version": "1.0.0", **POLICY, "horizon_sessions": 21,
            "meaning": "P(at least one of the next 21 SPY adjusted closes is strictly below 0.93 times the decision-date adjusted close)",
            "simple_interface": "predict_crisis(date) -> Python float in [0,1] after configure_inference(trusted=True)",
            "required_details": ["prediction_date", "p_crisis", "horizon_sessions", "model_name", "model_version",
                                 "feature_variant", "calibration_method", "prediction_origin", "training_cutoff", "decision_time", "status"],
            "decision_timezone": "America/New_York", "decision_local_hour": "17:00",
            "historical_origin": "historical_oof", "blind_origin": "blind_holdout_inference",
            "historical_policy": "saved annually refitted purged OOF only; refuse missing historical dates",
            "blind_policy": "frozen pre-2023 LR; label-free features; no holdout performance evaluation",
            "timing": "retrospective reconstruction, not evidence of contemporaneous publication",
            "training_cutoff_definition": "last training feature/decision date; training_label_end is reported separately",
            "monthly_policy": "historical: last admissible OOF session; blind: every calendar session of a fully covered month required",
            "trust": "joblib can execute code; load only artifacts from a trusted source; hashes are integrity checks, not authentication",
            "limitations": ["provisional candidate selected on validation; no statistical superiority established",
                            "raw probabilities are not certified calibrated; calibration none",
                            "adjusted-close snapshots may be revised; historical receipt by 17:00 is assumed",
                            "historical OOF and frozen blind policies are different",
                            "no allocation, investment recommendation or backtest"]}


@dataclass(frozen=True)
class PredictionDetails:
    prediction_date: str
    p_crisis: float
    horizon_sessions: int
    model_name: str
    model_version: str
    feature_variant: str
    calibration_method: str
    prediction_origin: str
    training_cutoff: str
    decision_time: str
    status: str
    training_label_end: str
    decision_timezone: str = "America/New_York"
    publication_status: str = "retrospective_not_contemporaneously_published"
    performance_evaluated: bool = False


class ML1Predictor:
    def __init__(self, model, manifest, oof, feature_source):
        self.model, self.manifest, self.oof, self.feature_source = model, manifest, oof, feature_source

    def predict_details(self, date):
        stamp = parse_prediction_date(date)
        if stamp < FINAL_TEST_START:
            for path, digest in self.manifest.get("source_sha256", {}).items():
                if Path(path).name in {"ml_models_oof_predictions.parquet", "training_metadata.json"} and sha256(_resolve(path)) != digest:
                    raise DataValidationError("Historical OOF source changed since loading")
            if stamp not in self.oof.index:
                raise DataValidationError("No admissible historical OOF for this date; frozen in-sample prediction forbidden")
            row = self.oof.loc[stamp]
            if int(row.validation_year) != stamp.year or row.fold != f"wf_{stamp.year}":
                raise DataValidationError("Historical prediction has invalid fold provenance")
            probability = float(row.predicted_probability)
            version = f"{row.configuration_id}/{row.fold}"
            origin = "historical_oof"
            cutoff, label_end = row.train_end, row.train_target_end_max
        else:
            X = self.feature_source.feature_at(stamp)
            probability = float(positive_probabilities(self.model, X)[0])
            version = self.manifest["model_id"]
            origin = "blind_holdout_inference"
            cutoff, label_end = self.manifest["training_end"], self.manifest["last_target_end"]
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise DataValidationError("p_crisis must be finite and in [0,1]")
        decision = decision_times(pd.DatetimeIndex([stamp])).iloc[0].tz_convert("America/New_York")
        return PredictionDetails(str(stamp.date()), probability, 21, "logistic_regression", version, "core", "none",
                                 origin, str(pd.Timestamp(cutoff).date()), decision.isoformat(), "provisional",
                                 str(pd.Timestamp(label_end).date()), performance_evaluated=origin == "historical_oof")

    def predict_crisis(self, date):
        return self.predict_details(date).p_crisis


def monthly_oof_probabilities(predictor):
    dates = predictor.oof.index.to_series().groupby(predictor.oof.index.to_period("M")).max()
    expected = pd.period_range(predictor.oof.index.min(), predictor.oof.index.max(), freq="M")
    if list(dates.index) != list(expected):
        raise DataValidationError("Missing historical OOF month; no fabricated month-end")
    return pd.DataFrame([{**asdict(predictor.predict_details(date)), "month": str(month)} for month, date in dates.items()])


def monthly_blind_probabilities(predictor):
    features = predictor.feature_source.features
    available = features.index[features.index >= FINAL_TEST_START]
    if available.empty: return pd.DataFrame(columns=[*PredictionDetails.__dataclass_fields__, "month"])
    first = available[0].to_period("M").start_time
    end = available[-1].to_period("M").end_time.normalize()
    calendar = xcals.get_calendar("ARCX", start=first, end=end)
    sessions = calendar.sessions.tz_localize(None)
    rows = []
    for month in pd.period_range(first, end, freq="M"):
        expected = sessions[sessions.to_period("M") == month]
        if expected.empty or not expected.isin(available).all(): continue
        # Month end must be an observed full month, not the last row of a partial month.
        detail = predictor.predict_details(expected[-1])
        rows.append({**asdict(detail), "month": str(month)})
    return pd.DataFrame(rows, columns=[*PredictionDetails.__dataclass_fields__, "month"])


def create_release(output_dir=DEFAULT_RELEASE, *, dataset_dir=DEFAULT_OUTPUT, raw_dir=DEFAULT_RAW,
                   training_dir=DEFAULT_TRAINING, expected_rows=4329):
    """Single authorized frozen fit; no holdout labels or metrics, no overwriting."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists(): raise FileExistsError(f"Refusing existing inference release: {output_dir}")
    if output_dir.is_relative_to(PROJECT_ROOT):
        relative = output_dir.relative_to(PROJECT_ROOT)
        if (len(relative.parts) < 3 or relative.parts[:2] != ("data", "processed")
                or not relative.parts[2].startswith("ml1_inference_release")):
            raise DataValidationError("Inference artifacts must stay in ignored data/processed/ml1_inference_release* directories")
    dataset_dir, raw_dir, training_dir = map(Path, (dataset_dir, raw_dir, training_dir))
    data = load_development_data(dataset_dir, raw_dir/"SPY_raw.csv")
    oof, old_metadata = load_saved_oof(data, training_dir)
    source = FeatureSource(dataset_dir, raw_dir)
    if not np.allclose(source.features.loc[data.X.index].to_numpy(), data.X.to_numpy(), rtol=0, atol=0):
        raise DataValidationError("Development and inference feature projections differ")
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "exchange-calendars")}
    identity = {"policy": POLICY, "configuration": configuration_spec("logistic_regression"),
                "features": list(FEATURE_COLUMNS), "versions": versions,
                "training_implementation_sha256": {name: sha256(Path(__file__).parent/name) for name in ("inference.py", "train_ml1.py", "core.py")},
                "training_rows": len(data.X), "training_end": str(data.X.index[-1].date()),
                "last_target_end": str(data.target_info.target_end_session.max().date()),
                "X_sha256": _fingerprint(data.X), "y_pre2023_sha256": _fingerprint(data.y),
                "target_controls_sha256": _fingerprint(data.target_info)}
    model_id = "ml1_lr_core_pre2023_"+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    model, fit_warnings = fit_frozen_model(data, expected_rows=expected_rows)
    sources = {**source.fingerprints,
               _portable(training_dir/"ml_models_oof_predictions.parquet"): sha256(training_dir/"ml_models_oof_predictions.parquet"),
               _portable(training_dir/"training_metadata.json"): sha256(training_dir/"training_metadata.json")}
    manifest = {"manifest_version": "1.0.0", "model_id": model_id, **POLICY,
                "python": sys.version,
                "resolved_estimator_parameters": json.loads(json.dumps(model.get_params(deep=True), default=repr)),
                "configuration_id": configuration_id("logistic_regression"), "identity": identity,
                "training_rows": len(data.X), "training_start": str(data.X.index[0].date()),
                "training_end": str(data.X.index[-1].date()), "last_target_end": str(data.target_info.target_end_session.max().date()),
                "last_training_label_known_at": str(decision_times(pd.DatetimeIndex([data.target_info.target_end_session.max()])).iloc[0]),
                "decision_time": "17:00", "decision_timezone": "America/New_York", "horizon_sessions": 21,
                "threshold": .07, "strict_threshold": True, "calibration_selected": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
                "source_sha256": sources, "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "resources": {"dataset_dir": _portable(dataset_dir), "raw_dir": _portable(raw_dir), "training_dir": _portable(training_dir)},
                "folds": old_metadata["folds"], "fit_warnings": fit_warnings,
                "scaler": {"n_samples_seen": int(model.named_steps['scaler'].n_samples_seen_),
                           "mean": model.named_steps['scaler'].mean_.tolist(), "variance": model.named_steps['scaler'].var_.tolist()},
                "retrospective": True, "holdout_labels_loaded": False, "holdout_performance_evaluated": False}
    predictor = ML1Predictor(model, manifest, oof, source)
    historical = monthly_oof_probabilities(predictor)
    blind = monthly_blind_probabilities(predictor)
    if expected_rows == 4329 and len(historical) != 95:
        raise DataValidationError("Expected exactly 95 historical OOF months")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        joblib.dump(model, stage/RELEASE_FILES[0], compress=3)
        historical.to_csv(stage/"monthly_oof_probabilities.csv", index=False, float_format="%.17g")
        blind.to_csv(stage/"monthly_blind_probabilities.csv", index=False, float_format="%.17g")
        (stage/"integration_contract.json").write_text(json.dumps(integration_contract(), indent=2)+"\n")
        manifest["artifact_sha256"] = {name: sha256(stage/name) for name in RELEASE_FILES if name != "model_manifest.json"}
        manifest["monthly_oof_rows"], manifest["monthly_blind_rows"] = len(historical), len(blind)
        (stage/"model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+"\n")
        source.check_unchanged()
        if output_dir.exists(): raise FileExistsError(f"Output appeared during export: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return manifest


def load_inference(release_dir=DEFAULT_RELEASE, *, trusted=False):
    """Explicit trust is mandatory before deserializing joblib, including the default."""
    if trusted is not True: raise PermissionError("Only load trusted joblib sources; explicitly pass trusted=True")
    release_dir = Path(release_dir)
    manifest = json.loads((release_dir/"model_manifest.json").read_text())
    if ({key: manifest.get(key) for key in POLICY} != POLICY
            or tuple(manifest["identity"]["features"]) != FEATURE_COLUMNS
            or manifest["configuration_id"] != configuration_id("logistic_regression")
            or pd.Timestamp(manifest["training_end"]) >= FINAL_TEST_START
            or pd.Timestamp(manifest["last_target_end"]) >= FINAL_TEST_START):
        raise DataValidationError("Frozen artifact policy/schema/training boundary mismatch")
    identity_id = "ml1_lr_core_pre2023_"+hashlib.sha256(json.dumps(manifest["identity"], sort_keys=True).encode()).hexdigest()[:16]
    if manifest["model_id"] != identity_id:
        raise DataValidationError("Frozen model identity mismatch")
    if set(manifest.get("artifact_sha256", {})) != set(RELEASE_FILES)-{"model_manifest.json"}:
        raise DataValidationError("Release must fingerprint every model, contract and monthly artifact")
    for name, digest in manifest["artifact_sha256"].items():
        if Path(name).name != name or sha256(release_dir/name) != digest:
            raise DataValidationError(f"Release artifact fingerprint mismatch: {name}")
    for path, digest in manifest["source_sha256"].items():
        if sha256(_resolve(path)) != digest:
            raise DataValidationError(f"Release source fingerprint mismatch: {path}")
    for name, digest in manifest["code_sha256"].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise DataValidationError(f"Release implementation changed: {name}")
    for name, version in manifest["versions"].items():
        if importlib.metadata.version(name) != version:
            raise DataValidationError(f"Dependency version mismatch: {name}")
    model = joblib.load(release_dir/"logistic_regression_final.joblib")
    if (not isinstance(model, Pipeline) or list(model.named_steps) != ["scaler", "classifier"]
            or not isinstance(model.named_steps["scaler"], StandardScaler)
            or not isinstance(model.named_steps["classifier"], LogisticRegression)
            or tuple(model.feature_names_in_) != FEATURE_COLUMNS):
        raise DataValidationError("Unexpected frozen estimator structure")
    for name, value in configuration_spec("logistic_regression")["parameters"].items():
        if model.named_steps["classifier"].get_params()[name] != value:
            raise DataValidationError("Frozen estimator configuration mismatch")
    scaler = model.named_steps["scaler"]
    if (int(scaler.n_samples_seen_) != manifest["training_rows"]
            or not np.array_equal(scaler.mean_, manifest["scaler"]["mean"])
            or not np.array_equal(scaler.var_, manifest["scaler"]["variance"])):
        raise DataValidationError("Frozen scaler manifest mismatch")
    resources = {name: _resolve(path) for name, path in manifest["resources"].items()}
    for path in [resources["dataset_dir"]/"dataset_metadata.json", resources["dataset_dir"]/"features_ml1_inference.parquet",
                 resources["raw_dir"]/"SPY_raw.csv", resources["raw_dir"]/"IEF_raw.csv",
                 resources["training_dir"]/"ml_models_oof_predictions.parquet", resources["training_dir"]/"training_metadata.json"]:
        if _portable(path) not in manifest["source_sha256"] or sha256(path) != manifest["source_sha256"][_portable(path)]:
            raise DataValidationError("Resource path does not match the fingerprinted release source")
    original_metadata = json.loads((resources["training_dir"]/"training_metadata.json").read_text())
    if original_metadata["folds"] != manifest["folds"]:
        raise DataValidationError("Frozen release changed the original OOF fold plan")
    source = FeatureSource(resources["dataset_dir"], resources["raw_dir"])
    # Load historical OOF from a label-free file, checked against the saved fold plans.
    frame = pd.read_parquet(resources["training_dir"]/"ml_models_oof_predictions.parquet")
    _check_dates_and_probabilities(frame, "model", ("logistic_regression", "random_forest", "xgboost"))
    oof = frame.loc[frame.model.eq("logistic_regression")].sort_values("prediction_date").set_index("prediction_date")
    expected = pd.DatetimeIndex([date for fold in manifest["folds"] for date in fold["validation_dates"]])
    if not oof.index.equals(expected): raise DataValidationError("Release OOF date plan mismatch")
    for fold in manifest["folds"]:
        q = oof.loc[oof.validation_year.eq(fold["validation_year"])]
        if (not q.configuration_id.eq(manifest["configuration_id"]).all()
                or not q.train_end.eq(pd.Timestamp(fold["train_end"])).all()
                or not q.train_target_end_max.eq(pd.Timestamp(fold["train_target_end_max"])).all()):
            raise DataValidationError("Release OOF training provenance mismatch")
    return ML1Predictor(model, manifest, oof, source)


def configure_inference(release_dir=DEFAULT_RELEASE, *, trusted=False):
    global _DEFAULT_PREDICTOR
    _DEFAULT_PREDICTOR = load_inference(release_dir, trusted=trusted)
    return _DEFAULT_PREDICTOR


def predict_crisis(date):
    if _DEFAULT_PREDICTOR is None:
        raise RuntimeError("Call configure_inference(release_dir, trusted=True) before predict_crisis")
    return float(_DEFAULT_PREDICTOR.predict_crisis(date))


def predict_crisis_details(date):
    if _DEFAULT_PREDICTOR is None:
        raise RuntimeError("Call configure_inference(release_dir, trusted=True) before prediction")
    return asdict(_DEFAULT_PREDICTOR.predict_details(date))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    if branch != "feature/ml1-crisis-detection": parser.exit(2, "Expected feature/ml1-crisis-detection\n")
    manifest = create_release(args.output_dir, dataset_dir=args.dataset_dir, raw_dir=args.raw_dir, training_dir=args.training_dir)
    print(json.dumps({key: manifest[key] for key in ["model_id", "training_rows", "training_end", "last_target_end", "monthly_oof_rows", "monthly_blind_rows"]}, indent=2))
    print("No holdout labels read or performance calculated. Policy: provisional, raw, core.")


if __name__ == "__main__":
    main()
