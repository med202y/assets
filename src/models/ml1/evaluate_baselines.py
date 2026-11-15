"""Paired walk-forward evaluation of two baselines, never the 2023+ holdout."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .baselines import (
    HistoricalFrequencyBaseline, VixThresholdBaseline, validate_development_labels,
    VIX_THRESHOLD, SMOOTHING_ALPHA, MIN_GROUP_SIZE,
)
from .core import DataValidationError
from .indicator_validation import (
    FeatureVariantData, build_feature_variant_folds, get_feature_variant_fold_xy,
    load_feature_variant,
)
from .market_indicators import DEFAULT_VIX_OUTPUT
from .pipeline import DEFAULT_OUTPUT, DEFAULT_RAW, PROJECT_ROOT
from .time_validation import FINAL_TEST_START, LABEL, WalkForwardConfig

DEFAULT_RESULTS = PROJECT_ROOT / "reports/ml1/baselines"
RESULT_FILES = ("baseline_metrics.csv", "baseline_predictions.parquet", "baseline_comparison.md")
EVALUATION_VERSION = "0.1.0"


def probability_metrics(y: pd.Series, probability) -> dict:
    """AP is average_precision_score, not a trapezoidal PR integral.

    Conservative reporting policy: both AP and ROC-AUC are NA for a one-class
    validation. AP with no positives is not meaningful; an all-positive sample
    also cannot assess discrimination. Brier remains defined for either case.
    """
    validate_development_labels(y)
    probability = np.asarray(probability, dtype=float)
    if (probability.shape != (len(y),) or not np.isfinite(probability).all()
            or ((probability < 0) | (probability > 1)).any()):
        raise DataValidationError("Metric probabilities must align with y and belong to [0, 1]")
    truth = y.to_numpy(dtype=np.int8)
    positives = int(truth.sum())
    both_classes = 0 < positives < len(truth)
    return {
        "n_observations": len(truth), "n_positive": positives, "n_negative": len(truth)-positives,
        "observed_crisis_rate": float(truth.mean()), "mean_predicted_probability": float(probability.mean()),
        "average_precision": float(average_precision_score(truth, probability)) if both_classes else np.nan,
        "brier_score": float(brier_score_loss(truth, probability, pos_label=1, scale_by_half=True)),
        "roc_auc": float(roc_auc_score(truth, probability)) if both_classes else np.nan,
        "metric_status": "both_classes" if both_classes else
                         ("single_class_no_positives" if positives == 0 else "single_class_no_negatives"),
    }


def evaluate_single_fold(model, X_train, y_train, X_validation, y_validation):
    """Fit and predict BEFORE passing validation labels to scoring."""
    if (X_train.empty or X_validation.empty
            or not X_train.index.equals(y_train.index) or not X_validation.index.equals(y_validation.index)
            or X_train.index.intersection(X_validation.index).size
            or X_train.index[-1] >= X_validation.index[0]):
        raise DataValidationError("Fold train/validation dates must be aligned, disjoint and chronological")
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_validation)[:, 1]
    metrics = probability_metrics(y_validation, probability)
    return probability, metrics, model.training_summary_


@dataclass(frozen=True)
class BaselineEvaluation:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    comparison_plan: tuple[dict, ...]
    training_estimates: tuple[dict, ...]


def plan_comparison(core: FeatureVariantData, vix: FeatureVariantData, config: WalkForwardConfig):
    """Choose common validation dates from indices alone, before fitting/scoring."""
    if core.profile.get("name") != "core" or vix.profile.get("name") != "core_vix":
        raise DataValidationError("Baseline study requires explicit core and core_vix variants")
    if vix.profile.get("status") != "experimental" or vix.profile.get("lag_spy_sessions") != 1:
        raise DataValidationError("VIX experiment requires the experimental exactly-one-SPY-session lag")
    core_folds, vix_folds = build_feature_variant_folds(core, config), build_feature_variant_folds(vix, config)
    if [f.validation_year for f in core_folds] != [f.validation_year for f in vix_folds]:
        raise DataValidationError("Both variants must provide the same annual folds")
    plan = []
    for first, second in zip(core_folds, vix_folds, strict=True):
        core_dates = core.X.index[first.validation_indices]
        vix_dates = vix.X.index[second.validation_indices]
        common = core_dates.intersection(vix_dates).sort_values()
        if len(common) < config.min_validation_observations:
            raise DataValidationError(f"Fold {first.validation_year}: insufficient common validation dates")
        plan.append({
            "validation_year": first.validation_year,
            "core_validation_rows": len(core_dates), "vix_validation_rows": len(vix_dates),
            "common_validation_rows": len(common),
            "common_dates": common.strftime("%Y-%m-%d").tolist(),
            "core_only_dates": core_dates.difference(vix_dates).strftime("%Y-%m-%d").tolist(),
            "vix_only_dates": vix_dates.difference(core_dates).strftime("%Y-%m-%d").tolist(),
            "n_train_core": len(first.train_indices), "n_train_vix": len(second.train_indices),
            "purged_core": len(first.purged_train_indices), "purged_vix": len(second.purged_train_indices),
        })
    return core_folds, vix_folds, tuple(plan)


def evaluate_baselines(
    core: FeatureVariantData | None = None, vix: FeatureVariantData | None = None,
    config: WalkForwardConfig | None = None, *,
    base_dir: Path = DEFAULT_OUTPUT, variant_dir: Path = DEFAULT_VIX_OUTPUT, raw_dir: Path = DEFAULT_RAW,
) -> BaselineEvaluation:
    config = config or WalkForwardConfig()
    core = core if core is not None else load_feature_variant("core", base_dir=base_dir, raw_dir=raw_dir)
    vix = vix if vix is not None else load_feature_variant(
        "core_vix", allow_experimental=True, base_dir=base_dir, variant_dir=variant_dir, raw_dir=raw_dir,
    )
    core_folds, vix_folds, plan = plan_comparison(core, vix, config)
    rows, prediction_frames, estimates = [], [], []
    specs = [
        ("historical_frequency", core, core_folds, False, HistoricalFrequencyBaseline),
        ("vix_threshold_25", vix, vix_folds, True, VixThresholdBaseline),
    ]
    for name, data, folds, experimental, factory in specs:
        for fold, alignment in zip(folds, plan, strict=True):
            X_train, y_train, X_val, y_val = get_feature_variant_fold_xy(data, fold)
            common = pd.DatetimeIndex(alignment["common_dates"])
            X_val, y_val = X_val.loc[common], y_val.loc[common]
            # Check outcome identity only for scoring consistency, never parameter selection.
            reference_y = core.core.y.loc[common]
            pd.testing.assert_series_equal(y_val, reference_y)
            model = factory()
            probability, scores, training = evaluate_single_fold(model, X_train, y_train, X_val, y_val)
            parameters = json.dumps(model.get_params(), sort_keys=True)
            training_json = json.dumps(training, sort_keys=True)
            fold_id = f"wf_{fold.validation_year}"
            shared = {"baseline": name, "feature_variant": data.profile["name"], "experimental": experimental}
            rows.append({
                **shared, "period_type": "fold", "fold": fold_id, "validation_year": fold.validation_year,
                "validation_start": str(common[0].date()), "validation_end": str(common[-1].date()),
                "n_train": len(X_train), "baseline_parameters": parameters, **scores,
            })
            prediction_frames.append(pd.DataFrame({
                "prediction_date": common, "fold": fold_id, "validation_year": fold.validation_year,
                **shared, "predicted_probability": probability,
                "observed_target": y_val.to_numpy(dtype=np.int8), "baseline_parameters": parameters,
                "training_estimates": training_json,
                "vix_lag_spy_sessions": 1 if experimental else pd.NA,
                "train_start": fold.train_start, "train_end": fold.train_end,
                "train_target_end_max": fold.train_target_end_max,
            }))
            estimates.append({**shared, "fold": fold_id, **training})
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["baseline", "prediction_date"], kind="stable"
    ).reset_index(drop=True)
    if ((predictions.prediction_date >= FINAL_TEST_START).any()
            or predictions.duplicated(["baseline", "prediction_date"]).any()):
        raise DataValidationError("OOF export must have unique pre-2023 dates per baseline")
    for name, group in predictions.groupby("baseline", sort=True):
        y = pd.Series(group.observed_target.to_numpy(),
                      index=pd.DatetimeIndex(group.prediction_date), name=LABEL)
        rows.append({
            "baseline": name, "feature_variant": group.feature_variant.iloc[0],
            "experimental": bool(group.experimental.iloc[0]), "period_type": "pooled_oof", "fold": "pooled_oof",
            "validation_year": pd.NA, "validation_start": str(y.index[0].date()),
            "validation_end": str(y.index[-1].date()), "n_train": pd.NA,
            "baseline_parameters": group.baseline_parameters.iloc[0],
            **probability_metrics(y, group.predicted_probability.to_numpy()),
        })
    metrics = pd.DataFrame(rows)
    return BaselineEvaluation(metrics, predictions, plan, tuple(estimates))


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append("NaN" if pd.isna(value) else f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def comparison_markdown(result: BaselineEvaluation) -> str:
    lines = [
        "**Comparaison des baselines ML 1 — développement uniquement**", "",
        "VIX expérimental, non certifié point-in-time. Aucun label ou score 2023+.", "",
        "**Protocole fixé avant les scores**", "",
        f"VIX < {VIX_THRESHOLD:g} / VIX >= {VIX_THRESHOLD:g}; Beta({SMOOTHING_ALPHA:g}, {SMOOTHING_ALPHA:g}); "
        f"repli global si groupe < {MIN_GROUP_SIZE} observations. Fréquences calculées sur train uniquement.", "",
        "**Alignement décidé avant estimation**", "",
    ]
    for item in result.comparison_plan:
        lines.append(f"- {item['validation_year']} : core {item['core_validation_rows']}, VIX {item['vix_validation_rows']}, "
                     f"commun {item['common_validation_rows']}; dates core seules {item['core_only_dates']}; "
                     f"dates VIX seules {item['vix_only_dates']}; train core/VIX "
                     f"{item['n_train_core']}/{item['n_train_vix']}; purge {item['purged_core']}/{item['purged_vix']}.")
    lines += ["", "Les deux baselines sont évaluées sur les mêmes dates communes. Les entraînements restent ceux de leurs folds purgés.",
              "", "**Résultats par fold**", ""]
    fold_metrics = result.metrics.loc[result.metrics.period_type == "fold"]
    lines.append(_markdown_table(fold_metrics, ["baseline", "validation_year", "n_observations", "n_positive", "n_negative",
                                               "observed_crisis_rate", "mean_predicted_probability", "average_precision", "brier_score", "roc_auc"]))
    lines += ["", "**Prédictions hors échantillon concaténées**", ""]
    lines.append(_markdown_table(result.metrics.loc[result.metrics.period_type == "pooled_oof"],
                                  ["baseline", "n_observations", "n_positive", "n_negative", "observed_crisis_rate",
                                   "mean_predicted_probability", "average_precision", "brier_score", "roc_auc"]))
    lines += ["", "AP est average_precision_score, pas une intégrale trapézoïdale. Brier plus faible = meilleur score probabiliste.",
              "AP et ROC-AUC sont NaN si une classe manque, par politique prudente de comparaison ; Brier reste défini.",
              "Les fenêtres de 21 séances se chevauchent : ni observations ni folds ne sont supposés indépendants.",
              "La fréquence est constante à l’intérieur de chaque fold ; son classement agrégé peut refléter les différences de fréquence entre années.",
              "Aucun ajustement sur validation, recherche de seuil, calibrateur, intervalle de confiance ou preuve de capacité prédictive.",
              "", "**Traçabilité**", "",
              f"Évaluation {EVALUATION_VERSION}; Python {platform.python_version()}; pandas {pd.__version__}; "
              f"NumPy {np.__version__}; scikit-learn {sklearn.__version__}.", "",
              "Les prédictions stockent paramètres fixes, estimations train, dates et références des folds.", ""]
    return "\n".join(lines)


def export_evaluation(result: BaselineEvaluation, output_dir: Path = DEFAULT_RESULTS) -> None:
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(PROJECT_ROOT) and not output_dir.is_relative_to(PROJECT_ROOT/"reports/ml1"):
        raise DataValidationError("Repository baseline results must stay under reports/ml1/")
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}; choose a new results directory")
    if (result.predictions.prediction_date >= FINAL_TEST_START).any():
        raise DataValidationError("No 2023+ prediction may be exported")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        result.metrics.to_csv(stage/"baseline_metrics.csv", index=False, na_rep="NaN")
        result.predictions.to_parquet(stage/"baseline_predictions.parquet", engine="pyarrow", index=False)
        (stage/"baseline_comparison.md").write_text(comparison_markdown(result), encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during export: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VIX_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.exit(2, f"Output exists: {args.output_dir}; choose a new results directory\n")
    result = evaluate_baselines(base_dir=args.base_dir, variant_dir=args.variant_dir, raw_dir=args.raw_dir)
    export_evaluation(result, args.output_dir)
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
