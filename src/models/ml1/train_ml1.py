"""Fixed core-only walk-forward training. No scoring, selection or calibration."""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import validate_development_features, validate_development_labels
from .core import DataValidationError
from .pipeline import DEFAULT_OUTPUT, DEFAULT_RAW, PROJECT_ROOT, sha256
from .time_validation import (
    FEATURE_COLUMNS, FINAL_TEST_START, TRAIN_START, DevelopmentData, WalkForwardConfig,
    WalkForwardFold, assert_fold_is_safe, build_walk_forward_folds, get_fold_xy,
    load_development_data,
)

TRAINING_VERSION = "0.1.0"
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
DEFAULT_TRAINING_OUTPUT = PROJECT_ROOT / "data/processed/ml1_training"
TRAINING_FILES = ("ml_models_oof_predictions.parquet", "training_metadata.json")
PREDICTION_COLUMNS = (
    "prediction_date", "validation_year", "fold", "model", "feature_variant",
    "predicted_probability", "configuration_id", "train_start", "train_end",
    "train_target_end_max", "latest_train_label_known_at", "first_validation_decision",
    "n_train", "n_purged",
)


def configuration_spec(name: str) -> dict:
    """Fresh fixed specification: no tuning or validation-dependent defaults."""
    specs = {
        "logistic_regression": {
            "preprocessing": "StandardScaler", "parameters": {
                "C": 1.0, "max_iter": 2000, "solver": "lbfgs", "class_weight": None,
            },
        },
        "random_forest": {
            "preprocessing": None, "parameters": {
                "n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt",
                "random_state": 42, "n_jobs": 1, "class_weight": None,
            },
        },
        "xgboost": {
            "preprocessing": None, "parameters": {
                "n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": "binary:logistic", "eval_metric": "logloss",
                "tree_method": "hist", "random_state": 42, "n_jobs": 1,
                "scale_pos_weight": 1,
            },
        },
    }
    if name not in specs:
        raise ValueError(f"Unknown fixed model: {name}")
    return {"model": name, "feature_variant": "core", "feature_columns": list(FEATURE_COLUMNS),
            "training_version": TRAINING_VERSION, **specs[name]}


def configuration_id(name: str) -> str:
    payload = json.dumps(configuration_spec(name), sort_keys=True, allow_nan=False)
    return f"{name}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def build_estimator(name: str):
    parameters = configuration_spec(name)["parameters"]
    if name == "logistic_regression":
        return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(**parameters))])
    if name == "random_forest":
        return RandomForestClassifier(**parameters)
    # Lazy import: a missing/native-broken XGBoost must not disable the other models.
    from xgboost import XGBClassifier
    return XGBClassifier(**parameters)


def xgboost_preflight(timeout: int = 60) -> dict:
    """Isolated native smoke test; a loader crash becomes a recorded failure."""
    script = '''
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import xgboost
from threadpoolctl import threadpool_info
from src.models.ml1.train_ml1 import build_estimator
X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]*8)
y=np.tile([0,0,1,1],8)
model=build_estimator("xgboost")
model.fit(X,y)
p=model.predict_proba(X[:4])
if p.shape!=(4,2) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
    raise RuntimeError("Invalid synthetic XGBoost probabilities")
libraries=[]
for path in [Path(xgboost.__file__).parent/"lib/libxgboost.dylib",
             Path(sys.prefix)/"lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib"]:
    libraries.append({"path":str(path),"exists":path.is_file(),
                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None})
print(json.dumps({"status":"passed","version":xgboost.__version__,"machine":platform.machine(),
                  "synthetic_rows":len(X),"probability_shape":list(p.shape),
                  "libraries":libraries,"threadpools":threadpool_info()}))
'''
    try:
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode:
            return {"status": "failed", "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr}
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        evidence["stderr"] = result.stderr
        return evidence
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}


def validate_training_inputs(X_train, y_train, X_validation) -> None:
    for X in (X_train, X_validation):
        validate_development_features(X, FEATURE_COLUMNS)
        if X.empty or (X.index < TRAIN_START).any():
            raise DataValidationError("Training study requires nonempty dates within 2005–2022")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise DataValidationError("Core features must be numeric")
    if not isinstance(y_train, pd.Series) or not X_train.index.equals(y_train.index):
        raise DataValidationError("Training features and labels must have identical dates")
    validate_development_labels(y_train)
    if set(y_train.unique()) != {0, 1}:
        raise DataValidationError("Supervised training requires both classes")
    if (X_train.index.intersection(X_validation.index).size
            or X_train.index[-1] >= X_validation.index[0]):
        raise DataValidationError("Train must be strictly earlier than validation")


def positive_probabilities(estimator, X_validation: pd.DataFrame) -> np.ndarray:
    """Extract class 1 by classes_, never assume a reversed class ordering."""
    matrix = np.asarray(estimator.predict_proba(X_validation), dtype=float)
    classes = np.asarray(estimator.classes_)
    if (matrix.shape != (len(X_validation), 2) or classes.shape != (2,)
            or set(classes.tolist()) != {0, 1}
            or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any()
            or not np.allclose(matrix.sum(axis=1), 1., rtol=0, atol=1e-6)):
        raise DataValidationError("Invalid binary predict_proba output")
    return matrix[:, int(np.flatnonzero(classes == 1)[0])].copy()


@dataclass(frozen=True)
class FoldFit:
    estimator: object
    probabilities: np.ndarray
    warnings: tuple[dict, ...]


def fit_predict_fold(name: str, data: DevelopmentData, fold: WalkForwardFold) -> FoldFit:
    """Reusable fitted fold estimator and probabilities; no y_validation argument."""
    assert_fold_is_safe(data, fold)
    X_train, y_train, X_validation, _ = get_fold_xy(data, fold)
    validate_training_inputs(X_train, y_train, X_validation)
    estimator = build_estimator(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X_train, y_train.to_numpy(dtype=np.int8))
        probabilities = positive_probabilities(estimator, X_validation)
    messages = tuple({"category": type(w.message).__name__, "message": str(w.message)} for w in caught)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(f"Fixed {name} configuration did not converge: {messages}")
    return FoldFit(estimator, probabilities, messages)


def _table_fingerprint(table) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()).hexdigest()


def _parameter_metadata(parameters: dict) -> dict:
    # XGBoost's default missing=np.nan is configuration, not a missing prediction.
    # Preserve it explicitly without producing non-standard JSON NaN literals.
    return json.loads(json.dumps(parameters, default=repr),
                      parse_constant=lambda value: {"nonfinite_float": value})


def _git_context() -> dict:
    result = {}
    for name, arguments in {"branch": ["branch", "--show-current"],
                            "head": ["rev-parse", "HEAD"],
                            "status": ["status", "--short"]}.items():
        command = subprocess.run(["git", *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True)
        result[name] = command.stdout.strip() if command.returncode == 0 else command.stderr.strip()
    return result


@dataclass(frozen=True)
class TrainingRun:
    predictions: pd.DataFrame
    metadata: dict


def train_walk_forward(data: DevelopmentData | None = None, config: WalkForwardConfig | None = None,
                       *, dataset_dir: Path = DEFAULT_OUTPUT,
                       spy_csv: Path = DEFAULT_RAW / "SPY_raw.csv") -> TrainingRun:
    """Run fixed models on unchanged folds; preserve successes if a model fails."""
    data = data if data is not None else load_development_data(dataset_dir, spy_csv)
    config = config or WalkForwardConfig()
    # All folds are checked before any real fit. No custom splits or VIX adapter.
    folds = build_walk_forward_folds(data, config)
    for fold in folds:
        assert_fold_is_safe(data, fold)
        X_train, y_train, X_validation, _ = get_fold_xy(data, fold)
        validate_training_inputs(X_train, y_train, X_validation)
    preflight = xgboost_preflight()
    records, frames, resolved_parameters = [], [], {}
    for fold in folds:
        dates = data.X.index[fold.validation_indices]
        for name in MODEL_NAMES:
            record = {"model": name, "fold": f"wf_{fold.validation_year}",
                      "configuration_id": configuration_id(name), "attempted": False}
            if name == "xgboost" and preflight["status"] != "passed":
                records.append({**record, "status": "unavailable", "error": "XGBoost preflight failed"})
                continue
            record["attempted"] = True
            try:
                fitted = fit_predict_fold(name, data, fold)
                resolved_parameters[name] = fitted.estimator.get_params(deep=True)
                frames.append(pd.DataFrame({
                    "prediction_date": dates, "validation_year": fold.validation_year,
                    "fold": record["fold"], "model": name, "feature_variant": "core",
                    "predicted_probability": fitted.probabilities,
                    "configuration_id": record["configuration_id"],
                    "train_start": fold.train_start, "train_end": fold.train_end,
                    "train_target_end_max": fold.train_target_end_max,
                    "latest_train_label_known_at": fold.latest_train_label_known_at,
                    "first_validation_decision": fold.first_validation_decision,
                    "n_train": len(fold.train_indices), "n_purged": len(fold.purged_train_indices),
                }))
                records.append({**record, "status": "success", "n_predictions": len(dates),
                                "warnings": list(fitted.warnings)})
            except Exception as exc:
                records.append({**record, "status": "failed", "error_type": type(exc).__name__,
                                "error": str(exc)})
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["model", "prediction_date"], kind="stable").reset_index(drop=True) if frames else pd.DataFrame()
    dependencies = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "pytest", "exchange-calendars"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    metadata = {
        "training_version": TRAINING_VERSION, "feature_variant": "core",
        "feature_columns": list(FEATURE_COLUMNS), "target": data.contract.get("target", {
            "name": "y_crisis_21d", "operational_threshold": .07, "horizon_spy_sessions": 21}),
        "status": "complete" if all(r["status"] == "success" for r in records) else "partial",
        "expected_fits": len(folds)*len(MODEL_NAMES),
        "attempted_fits": sum(r["attempted"] for r in records),
        "successful_fits": sum(r["status"] == "success" for r in records),
        "records": records, "xgboost_preflight": preflight,
        "configurations": {name: {**configuration_spec(name), "id": configuration_id(name)} for name in MODEL_NAMES},
        "resolved_estimator_parameters": _parameter_metadata(resolved_parameters),
        "git": _git_context(),
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "dependencies": dependencies,
        "development_start": str(data.X.index[0].date()), "development_end": str(data.X.index[-1].date()),
        "development_rows": len(data.X), "boundary_excluded_rows": len(data.boundary_excluded_dates),
        "dataframe_sha256": {"X": _table_fingerprint(data.X), "y_pre2023": _table_fingerprint(data.y),
                             "target_controls": _table_fingerprint(data.target_info)},
        "dataset_output_sha256": data.contract.get("output_sha256", {}),
        "raw_sources": data.contract.get("sources", []),
        "source_code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "folds": [{**f.summary(), "index_sha256": f.index_sha256,
                   "validation_dates": data.X.index[f.validation_indices].strftime("%Y-%m-%d").tolist(),
                   "train_indices_sha256": hashlib.sha256(f.train_indices.tobytes()).hexdigest(),
                   "validation_indices_sha256": hashlib.sha256(f.validation_indices.tobytes()).hexdigest(),
                   "latest_train_label_known_at": str(f.latest_train_label_known_at),
                   "first_validation_decision": str(f.first_validation_decision)} for f in folds],
        "policy": {"final_test_start": "2023-01-01", "final_labels_loaded": False,
                   "validation_labels_used_for_fit": False, "performance_metrics_computed": False,
                   "early_stopping": False, "calibration": False, "hyperparameter_search": False,
                   "resampling": False, "final_full_development_fit": False,
                   "shuffle_between_train_and_validation": False, "missing_values": "reject; no imputation",
                   "rng_note": "RF/XGBoost internal stochastic sampling fixed at 42; chronological folds unchanged"},
    }
    run = TrainingRun(predictions, metadata)
    validate_training_run(run)
    return run


def validate_training_run(run: TrainingRun) -> None:
    """Validate exports against fold plans and successful fits, never outcomes."""
    p, meta = run.predictions, run.metadata
    if meta.get("feature_variant") != "core" or tuple(meta.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise DataValidationError("Only the ten-column core variant can be exported")
    successful = [r for r in meta["records"] if r["status"] == "success"]
    if not successful:
        if not p.empty:
            raise DataValidationError("Predictions without successful fits")
        return
    if p.empty or tuple(p.columns) != PREDICTION_COLUMNS:
        raise DataValidationError("Prediction output requires its exact schema, without labels or future prices")
    if (p.prediction_date >= FINAL_TEST_START).any():
        raise DataValidationError("2023+ predictions are forbidden")
    if (not p.feature_variant.eq("core").all() or p.duplicated(["model", "prediction_date"]).any()
            or not np.isfinite(p.predicted_probability).all()
            or not p.predicted_probability.between(0, 1).all()):
        raise DataValidationError("Invalid core OOF probabilities or duplicate dates")
    if set(zip(p.model, p.fold)) != {(r["model"], r["fold"]) for r in successful}:
        raise DataValidationError("Prediction groups differ from successful fits")
    fold_plan = {f"wf_{f['validation_year']}": f for f in meta["folds"]}
    for record in successful:
        group = p[(p.model == record["model"]) & (p.fold == record["fold"])]
        plan = fold_plan[record["fold"]]
        dates = pd.DatetimeIndex(group.prediction_date)
        expected = pd.DatetimeIndex(plan["validation_dates"])
        if not dates.equals(expected) or len(dates) != record["n_predictions"]:
            raise DataValidationError("OOF dates differ from unchanged fold dates")
        if (not group.validation_year.eq(plan["validation_year"]).all()
                or not group.configuration_id.eq(configuration_id(record["model"])).all()
                or not group.train_end.lt(dates.min()).all()
                or not group.train_target_end_max.lt(dates.min()).all()
                or not group.latest_train_label_known_at.lt(group.first_validation_decision).all()):
            raise DataValidationError("Invalid prediction temporal/configuration provenance")


def export_training(run: TrainingRun, output_dir: Path = DEFAULT_TRAINING_OUTPUT) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    validate_training_run(run)
    if run.predictions.empty:
        raise DataValidationError("No successful model predictions to export")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        run.predictions.to_parquet(stage/TRAINING_FILES[0], index=False)
        metadata = {**run.metadata, "prediction_sha256": sha256(stage/TRAINING_FILES[0]),
                    "prediction_rows": len(run.predictions)}
        (stage/TRAINING_FILES[1]).write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during export: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spy-csv", type=Path, default=DEFAULT_RAW/"SPY_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    args = parser.parse_args()
    if _git_context()["branch"] != "feature/ml1-crisis-detection":
        parser.exit(2, "Training requires branch feature/ml1-crisis-detection\n")
    if args.output_dir.exists():
        parser.exit(2, f"Output exists: {args.output_dir}; choose a new directory\n")
    run = train_walk_forward(dataset_dir=args.dataset_dir, spy_csv=args.spy_csv)
    export_training(run, args.output_dir)
    print(f"Successful real fold fits: {run.metadata['successful_fits']}/{run.metadata['expected_fits']}")
    print(f"OOF prediction rows: {len(run.predictions)}; status: {run.metadata['status']}")
    if run.metadata["status"] != "complete":
        for record in run.metadata["records"]:
            if record["status"] != "success":
                print(json.dumps(record, sort_keys=True))
        parser.exit(1, "Partial results preserved; inspect training_metadata.json\n")


if __name__ == "__main__":
    main()
