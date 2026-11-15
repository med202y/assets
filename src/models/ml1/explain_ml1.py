"""Historical LR reconstruction for interpretation only; no final model export."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import plotting_environment, validate_oof_probabilities
from .core import DataValidationError
from .pipeline import sha256
from .time_validation import FEATURE_COLUMNS, FINAL_TEST_START, assert_fold_is_safe, get_fold_xy
from .train_ml1 import configuration_id, fit_predict_fold

PROBABILITY_ATOL = 1e-10
PROBABILITY_RTOL = 1e-8
ADDITIVITY_ATOL = 1e-9


@dataclass(frozen=True)
class ExplanationResult:
    values: np.ndarray
    base_values: np.ndarray
    features: pd.DataFrame
    importance: pd.DataFrame
    metadata: dict


def verify_reconstruction(reconstructed, saved):
    actual, expected = np.asarray(reconstructed, dtype=float), np.asarray(saved, dtype=float)
    if (actual.shape != expected.shape or not np.isfinite(actual).all()
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, atol=PROBABILITY_ATOL, rtol=PROBABILITY_RTOL)):
        raise DataValidationError("Reconstructed Logistic Regression diverges from recorded OOF; interpretation stopped")
    return float(np.max(np.abs(actual-expected)))


def explain_fitted_pipeline(pipeline, X_train, X_validation):
    """Actual interventional LinearExplainer, background entirely purged train."""
    for X in (X_train, X_validation):
        if tuple(X.columns) != FEATURE_COLUMNS:
            raise DataValidationError("SHAP requires exactly the ordered ten core features")
        if (X.index >= FINAL_TEST_START).any() or not np.isfinite(X.to_numpy()).all():
            raise DataValidationError("SHAP excludes 2023+ and nonfinite features")
    if X_train.empty or X_validation.empty or X_train.index[-1] >= X_validation.index[0]:
        raise DataValidationError("SHAP background must be strictly earlier than validation")
    if list(pipeline.named_steps) != ["scaler", "classifier"]:
        raise DataValidationError("Unexpected Logistic Regression preprocessing pipeline")
    scaler, classifier = pipeline.named_steps["scaler"], pipeline.named_steps["classifier"]
    if (int(scaler.n_samples_seen_) != len(X_train)
            or not np.allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-12, rtol=1e-10)
            or not np.allclose(scaler.var_, X_train.var(axis=0, ddof=0), atol=1e-12, rtol=1e-10)):
        raise DataValidationError("Scaler was not fitted exclusively on the supplied training rows")
    background = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=FEATURE_COLUMNS)
    transformed = pd.DataFrame(scaler.transform(X_validation), index=X_validation.index, columns=FEATURE_COLUMNS)
    with plotting_environment():
        import shap
        # Disable Independent's default 100-row subsampling: use every train row.
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.LinearExplainer(classifier, masker)
        explanation = explainer(transformed)
    values = np.asarray(explanation.values, dtype=float)
    base = np.broadcast_to(np.asarray(explanation.base_values, dtype=float), (len(X_validation),)).copy()
    decisions = pipeline.decision_function(X_validation)
    if (values.shape != (len(X_validation), len(FEATURE_COLUMNS)) or not np.isfinite(values).all()
            or not np.isfinite(base).all()
            or not np.allclose(base+values.sum(axis=1), decisions, atol=ADDITIVITY_ATOL, rtol=0)):
        raise DataValidationError("SHAP log-odds additivity check failed")
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS, "mean_abs_shap_log_odds": np.abs(values).mean(axis=0),
        "mean_signed_shap_log_odds": values.mean(axis=0),
        "standardized_coefficient": classifier.coef_[0],
    }).sort_values("mean_abs_shap_log_odds", ascending=False, kind="stable").reset_index(drop=True)
    metadata = {"status": "success", "method": "shap.LinearExplainer with Independent masker",
                "shap_version": importlib.metadata.version("shap"), "output_scale": "raw model log-odds, not probability changes",
                "feature_dependence": "interventional; correlated-feature and non-causal interpretation limitations",
                "background_rows": len(background), "background_start": str(X_train.index[0].date()),
                "background_end": str(X_train.index[-1].date()), "background_from_train_only": True,
                "background_sha256": hashlib.sha256(pd.util.hash_pandas_object(background, index=True).to_numpy().tobytes()).hexdigest(),
                "explained_rows": len(X_validation), "validation_start": str(X_validation.index[0].date()),
                "validation_end": str(X_validation.index[-1].date()), "expected_log_odds": float(base[0]),
                "additivity_max_abs_error": float(np.max(np.abs(base+values.sum(axis=1)-decisions))),
                "shap_source_sha256": sha256(Path(__file__)),
                "interpretation_scope": "one historical fold only; no cross-year importance stability claim"}
    return ExplanationResult(values, base, X_validation.copy(), importance, metadata)


def reconstruct_and_explain(data, fold, saved_probabilities):
    assert_fold_is_safe(data, fold)
    validate_oof_probabilities(saved_probabilities)
    dates = data.X.index[fold.validation_indices]
    if not dates.isin(saved_probabilities.index).all():
        raise DataValidationError("Missing saved probabilities for reconstruction check")
    # Explicitly authorized historical refit only. No saved predictions are replaced.
    fitted = fit_predict_fold("logistic_regression", data, fold)
    difference = verify_reconstruction(fitted.probabilities, saved_probabilities.loc[dates])
    X_train, _, X_validation, _ = get_fold_xy(data, fold)
    explanation = explain_fitted_pipeline(fitted.estimator, X_train, X_validation)
    return ExplanationResult(explanation.values, explanation.base_values, explanation.features,
                             explanation.importance, {**explanation.metadata, "validation_year": fold.validation_year,
                             "base_configuration_id": configuration_id("logistic_regression"),
                             "probability_max_abs_error": difference, "probability_atol": PROBABILITY_ATOL,
                             "probability_rtol": PROBABILITY_RTOL,
                             "reconstruction_purpose": "interpretability only; no production model or OOF regeneration",
                             "fit_warnings": list(fitted.warnings)})


def export_explanation(result, directory):
    directory = Path(directory)
    paths = [directory/name for name in ("shap_summary.png", "shap_importance.png", "shap_feature_importance.csv")]
    if any(p.exists() for p in paths): raise FileExistsError("Refusing to overwrite SHAP output")
    if result.metadata.get("status") != "success":
        raise DataValidationError("No validated SHAP explanation to export")
    with plotting_environment() as plt:
        import shap
        year = result.metadata.get("validation_year", result.features.index[0].year)
        # Fixed visualization jitter only; explanation values themselves are deterministic.
        shap.summary_plot(result.values, result.features, show=False, plot_size=(11, 7), max_display=10,
                          rng=np.random.default_rng(42))
        plt.gcf().suptitle(f"SHAP Logistic Regression — fold {year}, log-odds", y=.99)
        plt.gcf().text(.5, .01, f"Validation {year} · background train purgé seulement · interprétation interventional\nPas de contribution directe en probabilité ni de stabilité globale démontrée ; test final 2023+ fermé", ha="center", fontsize=8)
        plt.gcf().tight_layout(rect=(0, .08, 1, .96)); plt.savefig(paths[0], dpi=160); plt.close()
        ordered = result.importance.iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(ordered.feature, ordered.mean_abs_shap_log_odds, color="#0072B2")
        ax.set(xlabel="Moyenne |SHAP| en log-odds", title=f"Importance SHAP — Logistic Regression, fold {year} uniquement")
        fig.text(.5, .02, f"{len(result.features)} séances de validation {year} · background purgé historique · features corrélées\nImportance descriptive d’un seul modèle ; aucune causalité ni stabilité interannuelle démontrée", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(paths[1], dpi=160); plt.close(fig)
    result.importance.to_csv(paths[2], index=False)
