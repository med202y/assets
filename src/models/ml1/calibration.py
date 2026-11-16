"""Exploratory past-OOF-only calibration; no selection or production refit."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .baselines import validate_development_labels
from .core import DataValidationError, decision_times, validate_index
from .evaluate_ml1 import evaluate_saved_predictions, markdown_table, probability_metrics
from .pipeline import PROJECT_ROOT, sha256
from .time_validation import FINAL_TEST_START, LABEL, assert_fold_is_safe, build_walk_forward_folds, load_development_data
from .train_ml1 import configuration_id

EPSILON = 1e-6
MIN_OBSERVATIONS = 200
MIN_POSITIVES = 10
MIN_NEGATIVES = 10
METHODS = ("raw", "platt", "isotonic", "historical_frequency")
DEFAULT_OUTPUT = PROJECT_ROOT/"reports/ml1/calibration"
RESULT_FILES = (
    "calibrated_predictions.parquet", "calibration_metrics.csv", "calibration_comparison.md",
    "calibration_curves.png", "brier_calibration_by_year.png", "shap_summary.png",
    "shap_importance.png", "shap_feature_importance.csv",
)


def calibration_spec(method):
    if method == "platt":
        return {"method": method, "input": "logit_of_clipped_probability", "epsilon": EPSILON,
                "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": None}
    if method == "isotonic":
        return {"method": method, "out_of_bounds": "clip", "y_min": 0., "y_max": 1., "increasing": True}
    if method in {"raw", "historical_frequency"}:
        return {"method": method, "transformation": "none"}
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_id(method):
    spec = {**calibration_spec(method), "min_observations": MIN_OBSERVATIONS,
            "min_positives": MIN_POSITIVES, "min_negatives": MIN_NEGATIVES}
    return method+"_"+hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def numeric_logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), EPSILON, 1-EPSILON)
    return np.log(p/(1-p)).reshape(-1, 1)


@dataclass(frozen=True)
class CalibrationInputs:
    probabilities: pd.Series
    historical_baseline: pd.Series
    data: object
    folds: tuple
    provenance: dict


def load_calibration_inputs():
    # Reuse all step-6 source/schema/fingerprint checks, without fitting anything.
    checked = evaluate_saved_predictions()
    data = load_development_data()
    folds = build_walk_forward_folds(data)
    sources = {}
    for name in ("logistic_regression", "historical_frequency"):
        frame = checked.predictions.loc[checked.predictions.approach == name]
        sources[name] = pd.Series(frame.predicted_probability.to_numpy(),
                                  index=pd.DatetimeIndex(frame.prediction_date), name=name)
    # Confirm the saved step-6 metrics too, never overwrite them.
    path = PROJECT_ROOT/"reports/ml1/evaluation/model_metrics.csv"
    stored = pd.read_csv(path)
    for name in ("logistic_regression", "historical_frequency"):
        left = checked.metrics.loc[checked.metrics.approach == name].reset_index(drop=True)
        right = stored.loc[stored.approach == name].reset_index(drop=True)
        for column in ("average_precision", "brier_score", "roc_auc", "n_observations", "mean_predicted_probability"):
            if len(left) != len(right) or not np.allclose(left[column], right[column], atol=1e-12, rtol=0, equal_nan=True):
                raise DataValidationError("Step-6 saved metric fingerprint/content mismatch")
    provenance = {**checked.provenance, "step6_metrics_sha256": sha256(path),
                  "source_code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                  "shap_version": importlib.metadata.version("shap")}
    return CalibrationInputs(sources["logistic_regression"], sources["historical_frequency"], data, folds, provenance)


def validate_oof_probabilities(probabilities):
    if not isinstance(probabilities, pd.Series):
        raise DataValidationError("Saved OOF probabilities must be a dated Series")
    validate_index(probabilities.index)
    if (probabilities.index >= FINAL_TEST_START).any() or (probabilities.index.year < 2015).any():
        raise DataValidationError("OOF calibration study excludes 2023+ and pre-2015 predictions")
    if not np.isfinite(probabilities.to_numpy(dtype=float)).all() or not probabilities.between(0, 1).all():
        raise DataValidationError("OOF probabilities must be finite and in [0,1]")


def admissible_history(probabilities, data, fold):
    """Select dates and target maturity before touching historical label values."""
    validate_oof_probabilities(probabilities)
    assert_fold_is_safe(data, fold)
    current = data.X.index[fold.validation_indices]
    if not current.isin(probabilities.index).all() or not probabilities.index.isin(data.y.index).all():
        raise DataValidationError("OOF dates must align with protected development dates")
    previous = probabilities.index[(probabilities.index.year < fold.validation_year)
                                   & (probabilities.index < current[0])]
    ends = data.target_info.loc[previous].target_end_session
    known_at = decision_times(pd.DatetimeIndex(ends)) if len(previous) else pd.Series(dtype="datetime64[ns, UTC]")
    eligible = (ends < current[0]).to_numpy() & (known_at < fold.first_validation_decision).to_numpy()
    dates = previous[eligible]
    # No current-year, boundary or future labels are inspected before this selection.
    y = data.y.loc[dates].copy()
    if len(y): validate_development_labels(y)
    return dates, y, {
        "validation_year": fold.validation_year, "first_validation_date": str(current[0].date()),
        "first_validation_decision": str(fold.first_validation_decision),
        "n_previous_oof": len(previous), "n_excluded_immature": int((~eligible).sum()),
        "n_calibration": len(dates), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        "calibration_start": str(dates.min().date()) if len(dates) else None,
        "calibration_end": str(dates.max().date()) if len(dates) else None,
        "target_end_max": str(ends.loc[dates].max().date()) if len(dates) else None,
        "latest_label_known_at": str(known_at.loc[eligible].max()) if len(dates) else None,
        "history_dates_sha256": hashlib.sha256(dates.as_unit("ns").asi8.tobytes()).hexdigest(),
    }


def support_status(summary):
    if summary["n_previous_oof"] == 0:
        return "not_calibrated_no_history"
    if (summary["n_calibration"] < MIN_OBSERVATIONS or summary["n_positive"] < MIN_POSITIVES
            or summary["n_negative"] < MIN_NEGATIVES):
        return "not_calibrated_insufficient_support"
    return "calibrated"


def calibrate_fold(probabilities, data, fold):
    dates, y, summary = admissible_history(probabilities, data, fold)
    status = support_status(summary)
    current = data.X.index[fold.validation_indices]
    raw = probabilities.loc[current].to_numpy(dtype=float)
    frames, diagnostics = [], []
    for method in ("platt", "isotonic"):
        prediction, actual_status = raw.copy(), status
        details = {**summary, "method": method, "calibration_id": calibration_id(method),
                   "support_status": status, "warnings": []}
        if status == "calibrated":
            previous = probabilities.loc[dates].to_numpy(dtype=float)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if method == "platt":
                    model = LogisticRegression(C=1., solver="lbfgs", max_iter=2000, class_weight=None)
                    model.fit(numeric_logit(previous), y.to_numpy(dtype=np.int8))
                    prediction = model.predict_proba(numeric_logit(raw))[:, int(np.flatnonzero(model.classes_ == 1)[0])]
                    details["slope"] = float(model.coef_[0, 0])
                    details["intercept"] = float(model.intercept_[0])
                else:
                    model = IsotonicRegression(y_min=0., y_max=1., increasing=True, out_of_bounds="clip")
                    model.fit(previous, y.to_numpy(dtype=np.int8))
                    prediction = model.predict(raw)
                    details["n_thresholds"] = len(model.X_thresholds_)
                details["warnings"] = [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught]
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                prediction, actual_status = raw.copy(), "not_calibrated_convergence_failure"
        if (prediction.shape != raw.shape or not np.isfinite(prediction).all()
                or ((prediction < 0) | (prediction > 1)).any()):
            raise DataValidationError("Invalid calibrated probabilities")
        details["status"] = actual_status
        frames.append(pd.DataFrame({
            "prediction_date": current, "validation_year": fold.validation_year,
            "fold": f"wf_{fold.validation_year}", "method": method, "feature_variant": "core",
            "calibration_status": actual_status, "eligible_for_calibration": status == "calibrated",
            "raw_probability": raw, "predicted_probability": prediction,
            "base_configuration_id": configuration_id("logistic_regression"),
            "calibration_configuration_id": calibration_id(method),
            "calibration_start": pd.Timestamp(summary["calibration_start"]),
            "calibration_end": pd.Timestamp(summary["calibration_end"]),
            "calibration_target_end_max": pd.Timestamp(summary["target_end_max"]),
            "calibration_label_known_at_max": pd.Timestamp(summary["latest_label_known_at"]),
            "first_validation_decision": fold.first_validation_decision,
            "n_calibration": summary["n_calibration"], "n_calibration_positive": summary["n_positive"],
            "n_calibration_negative": summary["n_negative"], "n_excluded_immature": summary["n_excluded_immature"],
            "history_dates_sha256": summary["history_dates_sha256"],
            "final_test_excluded": True, "point_in_time_limitation": "adjusted-price core snapshot; historical availability assumptions retained",
        }))
        diagnostics.append(details)
    return pd.concat(frames, ignore_index=True), diagnostics


@dataclass(frozen=True)
class CalibrationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    history: tuple
    provenance: dict


def score_calibration(predictions, data):
    # A method failure must not give a different comparison sample to either method.
    eligible_dates = None
    for method in ("platt", "isotonic"):
        actual = pd.DatetimeIndex(predictions.loc[(predictions.method == method)
                                    & predictions.calibration_status.eq("calibrated"), "prediction_date"])
        eligible_dates = actual if eligible_dates is None else eligible_dates.intersection(actual)
    rows = []
    for scope, dates in [("all_validation", None), ("actually_calibrated_common", eligible_dates)]:
        for method in METHODS:
            group = predictions.loc[predictions.method == method]
            if dates is not None: group = group.loc[group.prediction_date.isin(dates)]
            for year in [*range(2015, 2023), None]:
                subset = group if year is None else group.loc[group.validation_year == year]
                if subset.empty: continue
                index = pd.DatetimeIndex(subset.prediction_date)
                y = data.y.loc[index]
                rows.append({"comparison_scope": scope, "method": method,
                             "period_type": "pooled_oof" if year is None else "fold", "validation_year": year,
                             "start": str(index.min().date()), "end": str(index.max().date()),
                             **probability_metrics(y, subset.predicted_probability.to_numpy())})
    return pd.DataFrame(rows)


def run_calibration(inputs=None):
    inputs = inputs if inputs is not None else load_calibration_inputs()
    validate_oof_probabilities(inputs.probabilities)
    validate_oof_probabilities(inputs.historical_baseline)
    if not inputs.probabilities.index.equals(inputs.historical_baseline.index):
        raise DataValidationError("Raw LR and historical baseline dates differ")
    expected = inputs.data.X.index[np.concatenate([f.validation_indices for f in inputs.folds])]
    if not expected.equals(inputs.probabilities.index):
        raise DataValidationError("Calibration must preserve all unchanged OOF dates")
    frames, history = [], []
    for fold in inputs.folds:
        frame, diagnostic = calibrate_fold(inputs.probabilities, inputs.data, fold)
        frames.append(frame); history.extend(diagnostic)
    calibrated = pd.concat(frames, ignore_index=True)
    template = calibrated.loc[calibrated.method == "platt"].copy()
    references = []
    for method in ("raw", "historical_frequency"):
        frame = template.copy()
        frame["method"] = method
        frame["calibration_status"] = "raw_reference" if method == "raw" else "baseline_reference"
        frame["calibration_configuration_id"] = calibration_id(method)
        frame["base_configuration_id"] = configuration_id("logistic_regression") if method == "raw" else "historical_frequency_train_rate"
        if method == "historical_frequency":
            frame["predicted_probability"] = inputs.historical_baseline.loc[pd.DatetimeIndex(frame.prediction_date)].to_numpy()
        else: frame["predicted_probability"] = frame.raw_probability
        references.append(frame)
    predictions = pd.concat([calibrated, *references], ignore_index=True).sort_values(
        ["method", "prediction_date"], kind="stable").reset_index(drop=True)
    validate_calibrated_predictions(predictions)
    return CalibrationResult(predictions, score_calibration(predictions, inputs.data), tuple(history),
                             {**inputs.provenance, "minimum_support": {"n": MIN_OBSERVATIONS, "positive": MIN_POSITIVES, "negative": MIN_NEGATIVES},
                              "configurations": {m: calibration_spec(m) for m in ("platt", "isotonic")},
                              "no_calibration_method_selected": True, "base_production_refits": 0,
                              "label_maturity": "target_end_session strictly before validation and known_at before first decision",
                              "distribution_shift": "base models differ across expanding-window OOF folds",
                              "calibration_source_sha256": sha256(Path(__file__))})


def validate_calibrated_predictions(p):
    if p.empty or set(p.method) != set(METHODS):
        raise DataValidationError("Four comparison methods required")
    if (p.prediction_date >= FINAL_TEST_START).any() or p.duplicated(["method", "prediction_date"]).any():
        raise DataValidationError("Duplicate or 2023+ calibrated prediction")
    reference = None
    for method, group in p.groupby("method"):
        dates = pd.DatetimeIndex(group.prediction_date)
        validate_index(dates)
        if reference is not None and not dates.equals(reference):
            raise DataValidationError("All four methods must have identical comparison dates")
        reference = dates
        if not group.validation_year.eq(dates.year).all():
            raise DataValidationError("Calibration year/date mismatch")
    for column in ("raw_probability", "predicted_probability"):
        if not np.isfinite(p[column]).all() or not p[column].between(0, 1).all():
            raise DataValidationError("Calibration probabilities must be finite and bounded")
    adjusted = p.loc[p.calibration_status.eq("calibrated")]
    if (not adjusted.calibration_end.lt(adjusted.prediction_date).all()
            or not adjusted.calibration_target_end_max.lt(adjusted.prediction_date).all()
            or not adjusted.calibration_label_known_at_max.lt(adjusted.first_validation_decision).all()):
        raise DataValidationError("Calibration history crosses validation boundary")


@contextmanager
def plotting_environment():
    """Use disposable writable caches; preserve user environment and global installs."""
    names = ("MPLCONFIGDIR", "XDG_CACHE_HOME")
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(prefix="ml1-calibration-plots-") as cache:
        for name in names: os.environ[name] = cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            yield plt
        finally:
            for name, value in previous.items():
                if value is None: os.environ.pop(name, None)
                else: os.environ[name] = value


def render_calibration_figures(result, data, output_dir):
    with plotting_environment() as plt:
        common = result.predictions.loc[result.predictions.calibration_status.eq("calibrated")]
        dates = pd.DatetimeIndex(common.prediction_date.unique()).sort_values()
        # The scoring function already intersects actual eligibility; repeat identically for plots.
        for method in ("platt", "isotonic"):
            dates = dates.intersection(pd.DatetimeIndex(common.loc[common.method == method, "prediction_date"]))
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        colors = {"raw": "#0072B2", "platt": "#D55E00", "isotonic": "#009E73", "historical_frequency": "#555555"}
        for ax, scope, selection in [(axes[0], "Toutes dates 2015–30/11/2022", None),
                                      (axes[1], "Dates effectivement calibrées communes", dates)]:
            for method in METHODS:
                group = result.predictions.loc[result.predictions.method == method]
                if selection is not None: group = group.loc[group.prediction_date.isin(selection)]
                if group.empty: continue
                y = data.y.loc[pd.DatetimeIndex(group.prediction_date)]
                true, predicted = calibration_curve(y, group.predicted_probability, n_bins=10, strategy="uniform")
                ax.plot(predicted, true, marker="o", color=colors[method], label=method)
            ax.plot([0, 1], [0, 1], ":", color="gray")
            ax.set(title=scope, xlabel="Probabilité moyenne par intervalle", ylabel="Fréquence observée", xlim=(0, 1), ylim=(0, 1))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.text(.5, .02, "10 intervalles fixes · descriptif OOF, non indépendant · repli 2015 inclus à gauche seulement\nFenêtres 21 séances chevauchantes ; pas de choix de méthode ; test final 2023+ fermé", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, 1)); fig.savefig(output_dir/"calibration_curves.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 6))
        for method in METHODS:
            rows = result.metrics.loc[result.metrics.comparison_scope.eq("all_validation")
                                      & result.metrics.period_type.eq("fold") & result.metrics.method.eq(method)]
            ax.plot(rows.validation_year, rows.brier_score, marker="o", color=colors[method], label=method)
        ax.set(title="Brier annuel — calibration chronologique exploratoire", xlabel="Année", ylabel="Brier (plus bas est préférable)", xticks=list(range(2015, 2023)))
        ax.legend(); ax.grid(alpha=.2)
        fig.text(.5, .02, "2015 : probabilités brutes conservées, aucune calibration · historique antérieur avec labels matures\nOOF 2015–30/11/2022 ; fenêtres chevauchantes ; test final 2023+ fermé", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .11, 1, 1)); fig.savefig(output_dir/"brier_calibration_by_year.png", dpi=160); plt.close(fig)


def comparison_markdown(result, explanation):
    pooled = result.metrics.loc[result.metrics.period_type.eq("pooled_oof")]
    return (
        "**Calibration ML1 exploratoire — aucun choix de méthode après scores**\n\n"
        "Platt : logit borné epsilon=1e-6, C=1, lbfgs, max_iter=2000. Isotonic : croissante, y_min=0, y_max=1, out_of_bounds=clip. "
        "Support fixé avant scores : 200 observations, 10 positifs, 10 négatifs. "
        "Historique exclusivement OOF d’années antérieures avec cibles et labels connus avant la première décision. "
        "2015 : not_calibrated_no_history ; aucune réussite de calibration attribuée à ce repli.\n\n"
        + markdown_table(pooled, ["comparison_scope", "method", "n_observations", "n_positive", "n_negative",
                                    "observed_crisis_rate", "mean_predicted_probability", "brier_score", "average_precision", "roc_auc"])
        + "\n\n**Historique admissible et paramètres appris**\n\n" + json.dumps(result.history, indent=2, ensure_ascii=False)
        + "\n\n**SHAP — fold 2022 uniquement**\n\n" + json.dumps(explanation.metadata, indent=2, ensure_ascii=False)
        + "\n\n**Limites**\n\nSélection préalable de Logistic Regression sur 2015–2022 : biais de sélection possible. "
        "Les modèles de base changent entre folds : décalage de distribution entre calibration historique et année courante. "
        "Platt peut inverser le classement si sa pente apprise sur l’historique est négative ; aucune contrainte ajoutée après scores. "
        "Isotonic peut créer des ex æquo et des probabilités extrêmes. Une AP élevée ne prouve pas la calibration. "
        "Fenêtres de 21 séances chevauchantes, aucune hypothèse d’indépendance ni test de significativité. "
        "Courbes de fiabilité descriptives, bins uniformes fixes, effectifs variables. "
        "SHAP interventional en log-odds sur un seul fold ; features corrélées, pas de causalité ni de stabilité interannuelle démontrée. "
        "Aucun label/prédiction 2023+, aucun modèle final sauvegardé.\n\n**Provenance**\n\n"
        + json.dumps(result.provenance, indent=2, ensure_ascii=False) + "\n"
    )


def export_results(result, explanation, data, output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite existing calibration output: {output_dir}")
    validate_calibrated_predictions(result.predictions)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        result.predictions.to_parquet(stage/"calibrated_predictions.parquet", index=False)
        result.metrics.to_csv(stage/"calibration_metrics.csv", index=False)
        (stage/"calibration_comparison.md").write_text(comparison_markdown(result, explanation), encoding="utf-8")
        render_calibration_figures(result, data, stage)
        if explanation.metadata.get("status") == "success":
            from .explain_ml1 import export_explanation
            export_explanation(explanation, stage)
        if output_dir.exists(): raise FileExistsError(f"Output appeared during export: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists(): shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    if branch != "feature/ml1-crisis-detection": parser.exit(2, "Expected feature/ml1-crisis-detection\n")
    if args.output_dir.exists(): parser.exit(2, f"Output exists: {args.output_dir}; choose a new directory\n")
    inputs = load_calibration_inputs()
    result = run_calibration(inputs)
    from .explain_ml1 import ExplanationResult, reconstruct_and_explain
    try:
        explanation = reconstruct_and_explain(inputs.data, inputs.folds[-1], inputs.probabilities)
    except Exception as exc:
        explanation = ExplanationResult(np.empty((0, 10)), np.empty(0), pd.DataFrame(), pd.DataFrame(),
                                        {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                         "missing_files": ["shap_summary.png", "shap_importance.png", "shap_feature_importance.csv"]})
    export_results(result, explanation, inputs.data, args.output_dir)
    print(result.metrics.loc[result.metrics.period_type.eq("pooled_oof")].to_string(index=False))
    print(json.dumps(explanation.metadata, indent=2, ensure_ascii=False))
    if explanation.metadata.get("status") != "success":
        parser.exit(1, "Calibration exported, SHAP unavailable; inspect comparison report\n")


if __name__ == "__main__":
    main()
