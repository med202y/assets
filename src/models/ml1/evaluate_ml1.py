"""Read-only evaluation of saved OOF probabilities; no estimator construction."""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from .core import DataValidationError
from .pipeline import DEFAULT_OUTPUT, DEFAULT_RAW, PROJECT_ROOT, sha256
from .time_validation import (
    FEATURE_COLUMNS, FINAL_TEST_START, LABEL, build_walk_forward_folds, load_development_data,
)

MODELS = ("logistic_regression", "random_forest", "xgboost")
BASELINES = ("historical_frequency", "vix_threshold_25")
APPROACHES = MODELS + BASELINES
DEFAULT_TRAINING = PROJECT_ROOT / "data/processed/ml1_training"
DEFAULT_BASELINES = PROJECT_ROOT / "reports/ml1/baselines"
DEFAULT_EVALUATION = PROJECT_ROOT / "reports/ml1/evaluation"
OUTPUT_FILES = ("model_metrics.csv", "model_comparison.md", "precision_recall_curves.png",
                "brier_by_year.png", "ap_by_year.png")
ML_COLUMNS = (
    "prediction_date", "validation_year", "fold", "model", "feature_variant",
    "predicted_probability", "configuration_id", "train_start", "train_end",
    "train_target_end_max", "latest_train_label_known_at", "first_validation_decision",
    "n_train", "n_purged",
)
BASELINE_COLUMNS = (
    "prediction_date", "fold", "validation_year", "baseline", "feature_variant", "experimental",
    "predicted_probability", "observed_target", "baseline_parameters", "training_estimates",
    "vix_lag_spy_sessions", "train_start", "train_end", "train_target_end_max",
)
METRIC_NAMES = ("average_precision", "brier_score", "roc_auc")
HISTORICAL_POOLED = {"historical_frequency": (.088309, .093446, .272661),
                     "vix_threshold_25": (.091453, .095820, .349705)}
SELECTION_POLICY = (
    "Rank the three ML configurations by pooled AP; inspect Brier and annual stability. "
    "A unique AP leader exceeding both baseline APs is only a provisional candidate. "
    "AP ties within 1e-12 or failure to exceed both baselines yield no unique candidate. "
    "No statistical superiority, tuning, calibration or final-test tie-breaking."
)


class AlignmentError(DataValidationError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__("OOF date mismatch; evaluation stopped before scores: " + json.dumps(diagnostic))


def probability_metrics(y: pd.Series, probability) -> dict:
    """Independent recalculation; single-class ranking scores remain NaN."""
    if not isinstance(y, pd.Series) or y.empty or not isinstance(y.index, pd.DatetimeIndex):
        raise DataValidationError("Metrics require nonempty dated labels")
    # Guard dates before comparing, converting or counting label values.
    if y.index.hasnans or (y.index >= FINAL_TEST_START).any():
        raise DataValidationError("2023+ labels are forbidden")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise DataValidationError("Labels must be complete and binary")
    p = np.asarray(probability, dtype=float)
    if p.shape != (len(y),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise DataValidationError("Probabilities must be finite, aligned and in [0,1]")
    truth = y.to_numpy(dtype=np.int8)
    positives = int(truth.sum())
    both = 0 < positives < len(y)
    return {
        "n_observations": len(y), "n_positive": positives, "n_negative": len(y)-positives,
        "observed_crisis_rate": float(truth.mean()), "mean_predicted_probability": float(p.mean()),
        "average_precision": float(average_precision_score(truth, p)) if both else np.nan,
        "brier_score": float(brier_score_loss(truth, p, pos_label=1, scale_by_half=True)),
        "roc_auc": float(roc_auc_score(truth, p)) if both else np.nan,
        "metric_status": "both_classes" if both else
                         ("single_class_no_positives" if positives == 0 else "single_class_no_negatives"),
        "ranking_metric_reason": "" if both else "A single class cannot assess two-class discrimination",
    }


def _fingerprint(table) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()).hexdigest()


def _check_dates_and_probabilities(frame, identity, expected_names):
    required = {identity, "prediction_date", "validation_year", "fold", "predicted_probability"}
    if not required.issubset(frame) or frame.empty:
        raise DataValidationError("Missing OOF prediction schema")
    dates = frame.prediction_date
    if (not pd.api.types.is_datetime64_any_dtype(dates) or dates.isna().any()
            or dates.dt.tz is not None or (dates >= FINAL_TEST_START).any()
            or (dates < pd.Timestamp("2015-01-01")).any()):
        raise DataValidationError("OOF predictions must have naive dates within 2015–2022")
    if set(frame[identity]) != set(expected_names):
        raise DataValidationError("Unexpected or missing model/baseline names")
    if frame.duplicated([identity, "prediction_date"]).any():
        raise DataValidationError("Duplicate OOF prediction per approach/date")
    if (not frame.validation_year.eq(dates.dt.year).all()
            or not frame.fold.eq("wf_" + frame.validation_year.astype(str)).all()):
        raise DataValidationError("Fold/year does not match prediction date")
    if not pd.api.types.is_numeric_dtype(frame.predicted_probability):
        raise DataValidationError("Probabilities must be numeric")
    if (not np.isfinite(frame.predicted_probability).all()
            or not frame.predicted_probability.between(0, 1).all()):
        raise DataValidationError("Probabilities must be finite and in [0,1]")


def align_predictions(ml: pd.DataFrame, baselines: pd.DataFrame, data, folds) -> tuple[pd.DataFrame, dict]:
    """Require the unchanged date plan before inspecting baseline label values."""
    _check_dates_and_probabilities(ml, "model", MODELS)
    _check_dates_and_probabilities(baselines, "baseline", BASELINES)
    expected = data.X.index[np.concatenate([f.validation_indices for f in folds])]
    diagnostic = {"expected_dates": len(expected), "approaches": {}}
    mismatch = False
    for source, key, names in ((ml, "model", MODELS), (baselines, "baseline", BASELINES)):
        for name in names:
            actual = pd.DatetimeIndex(source.loc[source[key] == name, "prediction_date"]).sort_values()
            missing, extra = expected.difference(actual), actual.difference(expected)
            diagnostic["approaches"][name] = {
                "n_dates": len(actual), "missing_dates": missing.strftime("%Y-%m-%d").tolist(),
                "extra_dates": extra.strftime("%Y-%m-%d").tolist(),
            }
            mismatch |= bool(len(missing) or len(extra))
    if mismatch:
        raise AlignmentError(diagnostic)
    parts = []
    for source, key, names in ((ml, "model", MODELS), (baselines, "baseline", BASELINES)):
        for name in names:
            group = source.loc[source[key] == name].sort_values("prediction_date").copy()
            dates = pd.DatetimeIndex(group.prediction_date)
            labels = data.y.reindex(dates)
            if labels.isna().any() or not (data.target_info.loc[dates].target_end_session < FINAL_TEST_START).all():
                raise DataValidationError("Missing or inadmissible pre-2023 target in label join")
            if key == "baseline":
                if not group.observed_target.isin([0, 1]).all() or not np.array_equal(
                    group.observed_target.to_numpy(), labels.to_numpy()
                ):
                    raise DataValidationError("Saved baseline labels differ from protected development labels")
                experimental = name == "vix_threshold_25"
                if (not group.experimental.eq(experimental).all()
                        or not group.feature_variant.eq("core_vix" if experimental else "core").all()
                        or (experimental and not group.vix_lag_spy_sessions.eq(1).all())):
                    raise DataValidationError("VIX baseline must remain EXPERIMENTAL with one-session lag")
            else:
                experimental = False
                if not group.feature_variant.eq("core").all():
                    raise DataValidationError("ML evaluation requires core-only predictions")
            parts.append(pd.DataFrame({
                "approach": name, "kind": "baseline" if key == "baseline" else "ml",
                "feature_variant": group.feature_variant.to_numpy(), "experimental": experimental,
                "prediction_date": dates, "validation_year": dates.year,
                "predicted_probability": group.predicted_probability.to_numpy(),
                "observed_target": labels.to_numpy(dtype=np.int8),
            }))
    return pd.concat(parts, ignore_index=True), diagnostic


def validate_training_provenance(ml, metadata, data, folds):
    if (tuple(ml.columns) != ML_COLUMNS or metadata.get("status") != "complete"
            or metadata.get("successful_fits") != 24 or metadata.get("prediction_rows") != len(ml)
            or metadata.get("feature_variant") != "core"
            or tuple(metadata.get("feature_columns", [])) != FEATURE_COLUMNS):
        raise DataValidationError("Saved ML schema/metadata differs from completed step 5")
    for key, table in {"X": data.X, "y_pre2023": data.y, "target_controls": data.target_info}.items():
        if metadata["dataframe_sha256"][key] != _fingerprint(table):
            raise DataValidationError(f"Training data fingerprint mismatch: {key}")
    if len(metadata["folds"]) != len(folds):
        raise DataValidationError("Training fold metadata differs from existing engine")
    records = metadata["records"]
    if (len(records) != 24 or {(r["model"], r["fold"]) for r in records} !=
            {(name, f"wf_{f.validation_year}") for name in MODELS for f in folds}
            or any(r["status"] != "success" for r in records)):
        raise DataValidationError("Incomplete or duplicate step-5 training records")
    for name in MODELS:
        spec = dict(metadata["configurations"][name])
        identifier = spec.pop("id")
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
        if identifier != f"{name}_{digest}" or not ml.loc[ml.model == name].configuration_id.eq(identifier).all():
            raise DataValidationError("Configuration identifier mismatch")
        if any(r["configuration_id"] != identifier for r in records if r["model"] == name):
            raise DataValidationError("Training record configuration mismatch")
    for fold, saved in zip(folds, metadata["folds"], strict=True):
        dates = data.X.index[fold.validation_indices]
        expected = {**fold.summary(), "index_sha256": fold.index_sha256,
                    "validation_dates": dates.strftime("%Y-%m-%d").tolist(),
                    "train_indices_sha256": hashlib.sha256(fold.train_indices.tobytes()).hexdigest(),
                    "validation_indices_sha256": hashlib.sha256(fold.validation_indices.tobytes()).hexdigest(),
                    "latest_train_label_known_at": str(fold.latest_train_label_known_at),
                    "first_validation_decision": str(fold.first_validation_decision)}
        if saved != expected:
            raise DataValidationError(f"Fold provenance mismatch: {fold.validation_year}")
        group = ml.loc[ml.fold == f"wf_{fold.validation_year}"]
        for column, value in {"train_start": fold.train_start, "train_end": fold.train_end,
                              "train_target_end_max": fold.train_target_end_max,
                              "latest_train_label_known_at": fold.latest_train_label_known_at,
                              "first_validation_decision": fold.first_validation_decision,
                              "n_train": len(fold.train_indices), "n_purged": len(fold.purged_train_indices)}.items():
            if not group[column].eq(value).all():
                raise DataValidationError(f"OOF temporal provenance mismatch: {column}")


def compute_metrics(predictions) -> pd.DataFrame:
    rows = []
    for name in APPROACHES:
        group = predictions.loc[predictions.approach == name]
        common = {"approach": name, "kind": group.kind.iloc[0],
                  "feature_variant": group.feature_variant.iloc[0], "experimental": bool(group.experimental.iloc[0])}
        for year in [*range(2015, 2023), None]:
            subset = group if year is None else group.loc[group.validation_year == year]
            dates = pd.DatetimeIndex(subset.prediction_date)
            y = pd.Series(subset.observed_target.to_numpy(), index=dates, name=LABEL)
            rows.append({**common, "period_type": "pooled_oof" if year is None else "fold",
                         "validation_year": year, "validation_start": str(dates.min().date()),
                         "validation_end": str(dates.max().date()),
                         **probability_metrics(y, subset.predicted_probability.to_numpy())})
    return pd.DataFrame(rows)


def verify_baseline_metrics(recalculated: pd.DataFrame, saved: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected_keys = {(name, "fold", year) for name in BASELINES for year in range(2015, 2023)}
    expected_keys |= {(name, "pooled_oof", None) for name in BASELINES}
    keys = [(r.baseline, r.period_type, None if pd.isna(r.validation_year) else int(r.validation_year))
            for r in saved.itertuples()]
    if len(keys) != 18 or set(keys) != expected_keys:
        raise DataValidationError("Historical baseline metric periods are incomplete or duplicated")
    for old in saved.itertuples():
        chosen = recalculated[(recalculated.approach == old.baseline) & (recalculated.period_type == old.period_type)]
        if old.period_type == "fold":
            chosen = chosen.loc[chosen.validation_year == old.validation_year]
        new = chosen.iloc[0]
        for column in [*METRIC_NAMES, "n_observations", "n_positive", "n_negative",
                       "observed_crisis_rate", "mean_predicted_probability"]:
            a, b = float(new[column]), float(getattr(old, column))
            agrees = (np.isnan(a) and np.isnan(b)) or abs(a-b) <= 1e-12
            if not agrees:
                raise DataValidationError(f"Baseline history mismatch: {old.baseline}/{old.validation_year}/{column}")
        if old.period_type == "pooled_oof":
            errors = {column: abs(float(new[column])-rounded)
                      for column, rounded in zip(METRIC_NAMES, HISTORICAL_POOLED[old.baseline], strict=True)}
            if any(error > 1e-6 for error in errors.values()):
                raise DataValidationError("Baseline aggregate differs from the step-4 rounded reference")
            rows.append({"baseline": old.baseline, "historical_csv_match": True, **errors})
    return pd.DataFrame(rows)


def provisional_recommendation(metrics) -> dict:
    pooled = metrics.loc[metrics.period_type == "pooled_oof"].set_index("approach")
    ranking = pooled.loc[list(MODELS)].sort_values(["average_precision", "brier_score"], ascending=[False, True])
    top, second = ranking.index[:2]
    ap = ranking.average_precision
    unique = pd.notna(ap.iloc[0]) and ap.iloc[0]-ap.iloc[1] > 1e-12
    beats = bool(ap.iloc[0] > pooled.loc[list(BASELINES)].average_precision.max())
    candidate = top if unique and beats else None
    annual = metrics.loc[metrics.period_type == "fold"].pivot(index="validation_year", columns="approach", values="average_precision")
    return {"policy": SELECTION_POLICY, "candidate": candidate, "status": "CANDIDAT PROVISOIRE" if candidate else "AUCUN CANDIDAT UNIQUE CONVAINCANT",
            "ml_ranking_by_pooled_ap": ranking.index.tolist(),
            "pooled_ap_gap_to_second_ml": float(ap.iloc[0]-ap.iloc[1]),
            "best_ml_brier": ranking.brier_score.idxmin(),
            "annual_ap_wins": {name: int((annual[name] == annual[list(MODELS)].max(axis=1)).sum()) for name in MODELS},
            "annual_two_class_periods": int(annual[list(MODELS)].notna().all(axis=1).sum()),
            "statistical_superiority_established": False,
            "selection_bias": "Selection uses 2015–2022 validation outcomes; this is not an unbiased final-test estimate."}


@dataclass(frozen=True)
class EvaluationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    alignment: dict
    baseline_verification: pd.DataFrame
    recommendation: dict
    provenance: dict


def evaluate_saved_predictions(*, training_dir=DEFAULT_TRAINING, baseline_dir=DEFAULT_BASELINES,
                               dataset_dir=DEFAULT_OUTPUT, spy_csv=DEFAULT_RAW/"SPY_raw.csv") -> EvaluationResult:
    training_dir, baseline_dir, dataset_dir = map(Path, (training_dir, baseline_dir, dataset_dir))
    prediction_path = training_dir/"ml_models_oof_predictions.parquet"
    metadata_path = training_dir/"training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if sha256(prediction_path) != metadata["prediction_sha256"]:
        raise DataValidationError("Saved ML prediction fingerprint mismatch")
    for name, digest in metadata["source_code_sha256"].items():
        if sha256(Path(__file__).parent/name) != digest:
            raise DataValidationError(f"Step-5 source fingerprint mismatch: {name}")
    for name, digest in metadata["dataset_output_sha256"].items():
        if sha256(dataset_dir/name) != digest:
            raise DataValidationError(f"Step-5 dataset fingerprint mismatch: {name}")
    if tuple(pq.ParquetFile(prediction_path).schema_arrow.names) != ML_COLUMNS:
        raise DataValidationError("Unexpected saved ML Parquet schema")
    baseline_path = baseline_dir/"baseline_predictions.parquet"
    if tuple(pq.ParquetFile(baseline_path).schema_arrow.names) != BASELINE_COLUMNS:
        raise DataValidationError("Unexpected saved baseline Parquet schema")
    committed_baselines = {}
    if baseline_dir.resolve() == DEFAULT_BASELINES.resolve():
        for name in ("baseline_predictions.parquet", "baseline_metrics.csv", "baseline_comparison.md"):
            path = baseline_dir/name
            relative = path.relative_to(PROJECT_ROOT)
            saved = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=PROJECT_ROOT)
            digest = hashlib.sha256(saved).hexdigest()
            if sha256(path) != digest:
                raise DataValidationError(f"Committed baseline fingerprint mismatch: {name}")
            committed_baselines[name] = digest
    # Inspect dates before reading baseline label values, then filter their read explicitly.
    ml = pd.read_parquet(prediction_path)
    baseline = pd.read_parquet(baseline_path, columns=[c for c in BASELINE_COLUMNS if c != "observed_target"])
    _check_dates_and_probabilities(ml, "model", MODELS)
    _check_dates_and_probabilities(baseline, "baseline", BASELINES)
    labels = pd.read_parquet(baseline_path, columns=["baseline", "prediction_date", "observed_target"],
                             filters=[("prediction_date", "<", FINAL_TEST_START)])
    baseline = baseline.merge(labels, on=["baseline", "prediction_date"], how="left", validate="one_to_one")
    # The step-3 loader is the sole source of truth for development targets.
    data = load_development_data(dataset_dir, Path(spy_csv))
    folds = build_walk_forward_folds(data)
    validate_training_provenance(ml, metadata, data, folds)
    joined, alignment = align_predictions(ml, baseline, data, folds)
    metrics = compute_metrics(joined)
    saved_metrics = pd.read_csv(baseline_dir/"baseline_metrics.csv")
    verified = verify_baseline_metrics(metrics, saved_metrics)
    # Recommendation is deliberately downstream of all baseline checks.
    recommendation = provisional_recommendation(metrics)
    paths = [prediction_path, metadata_path, baseline_path, baseline_dir/"baseline_metrics.csv",
             baseline_dir/"baseline_comparison.md", dataset_dir/"dataset_metadata.json"]
    provenance = {"input_sha256": {str(p): sha256(p) for p in paths},
                  "committed_baseline_sha256": committed_baselines,
                  "baseline_metrics_verified": True, "new_fits": 0,
                  "feature_columns": list(FEATURE_COLUMNS),
                  "dependencies": {name: importlib.metadata.version(name) for name in
                                   ["numpy", "pandas", "pyarrow", "scikit-learn", "matplotlib"]},
                  "configuration_ids": {name: metadata["configurations"][name]["id"] for name in MODELS}}
    return EvaluationResult(joined, metrics, alignment, verified, recommendation, provenance)


def markdown_table(frame, columns) -> str:
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value): cell = "NaN"
            elif isinstance(value, (float, np.floating)): cell = f"{value:.6f}"
            else: cell = str(value)
            cells.append(cell)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def comparison_markdown(result) -> str:
    annual = result.metrics.loc[result.metrics.period_type == "fold"]
    pooled = result.metrics.loc[result.metrics.period_type == "pooled_oof"]
    columns = ["approach", "n_observations", "n_positive", "n_negative", "observed_crisis_rate",
               "mean_predicted_probability", *METRIC_NAMES]
    return (
        "**ML1 — comparaison des probabilités OOF enregistrées, 2015–30 novembre 2022**\n\n"
        "Aucun nouvel entraînement, tuning ou calibration. VIX : **EXPÉRIMENTAL**, non certifié point-in-time.\n\n"
        + "**Agrégation OOF recalculée sur les prédictions individuelles**\n\n"
        + markdown_table(pooled, columns) + "\n\n**Résultats annuels**\n\n"
        + markdown_table(annual, ["validation_year", *columns, "metric_status"]) + "\n\n"
        + "NaN AP/ROC : une seule classe, discrimination non évaluable ; aucun remplacement par zéro ou 0,5.\n\n"
        + "**Méthode et candidat provisoire**\n\n" + SELECTION_POLICY + "\n\n"
        + json.dumps(result.recommendation, indent=2, ensure_ascii=False) + "\n\n"
        + interpretation_markdown(result) + "\n\n"
        + "L’AP mesure le classement ; le Brier évalue l’erreur quadratique des probabilités et ne suffit pas à isoler leur calibration. "
          "La ROC-AUC est complémentaire. Une AP supérieure ne garantit pas des probabilités calibrées. "
          "Les fenêtres de 21 séances se chevauchent ; aucun test de significativité fondé sur leur indépendance. "
          "Le candidat est sélectionné sur les validations : biais de sélection possible. "
          "Toute calibration ultérieure doit utiliser une séparation temporelle propre, sans calibrer une prédiction sur son propre label. "
          "Test final 2023+ fermé ; aucun modèle final sauvegardé.\n\n"
        + "**Alignement et vérifications des baselines**\n\n"
        + json.dumps(result.alignment, indent=2, ensure_ascii=False) + "\n\n"
        + markdown_table(result.baseline_verification, list(result.baseline_verification.columns)) + "\n\n"
        + "**Provenance**\n\n" + json.dumps(result.provenance, indent=2, ensure_ascii=False) + "\n"
    )


def interpretation_markdown(result) -> str:
    pooled = result.metrics.loc[result.metrics.period_type == "pooled_oof"].set_index("approach")
    candidate = result.recommendation["candidate"]
    if candidate is None:
        return "Aucun candidat unique convaincant selon la règle préétablie ; aucune sélection forcée."
    row = pooled.loc[candidate]
    historical = pooled.loc["historical_frequency"]
    return (
        f"**CANDIDAT PROVISOIRE pour l’étape 7 : {candidate}.** "
        f"Son AP agrégée ({row.average_precision:.6f}) est la plus élevée des trois modèles "
        f"et dépasse les deux baselines. Son Brier ({row.brier_score:.6f}) doit être comparé "
        f"à celui de la fréquence historique ({historical.brier_score:.6f}) ; "
        + ("il est moins bon. " if row.brier_score > historical.brier_score else "il est inférieur. ")
        + f"ROC-AUC : {row.roc_auc:.6f}. "
        f"Probabilité moyenne : {row.mean_predicted_probability:.2%}, contre une fréquence observée "
        f"de {row.observed_crisis_rate:.2%}. Ce décalage moyen ne remplace pas un diagnostic de calibration. "
        f"Victoires annuelles AP parmi les trois ML : {result.recommendation['annual_ap_wins']}, "
        f"sur {result.recommendation['annual_two_class_periods']} années à deux classes. "
        "La préférence est limitée au critère AP de validation ; la stabilité annuelle et la qualité "
        "des probabilités ne permettent pas de déclarer une supériorité universelle ou statistiquement établie. "
        "Aucune configuration n’est modifiée après observation des résultats."
    )


def render_figures(result, output_dir):
    # Keep the Matplotlib cache outside the repository and use a non-GUI backend.
    previous = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="ml1-evaluation-mpl-") as config:
        os.environ["MPLCONFIGDIR"] = config
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            colors = dict(zip(APPROACHES, ["#0072B2", "#D55E00", "#009E73", "#555555", "#CC79A7"], strict=True))
            labels = {name: name + (" [EXPÉRIMENTAL]" if name == "vix_threshold_25" else "") for name in APPROACHES}
            note = "Validation 2015–30/11/2022 · cibles de 21 séances chevauchantes · test final 2023+ fermé"
            fig, ax = plt.subplots(figsize=(11, 7))
            for name in APPROACHES:
                group = result.predictions.loc[result.predictions.approach == name]
                precision, recall, _ = precision_recall_curve(group.observed_target, group.predicted_probability)
                ax.step(recall, precision, where="post", color=colors[name],
                        linestyle="--" if name in BASELINES else "-", label=labels[name])
            prevalence = result.predictions.loc[result.predictions.approach == MODELS[0], "observed_target"].mean()
            ax.axhline(prevalence, color="#888888", linestyle=":", label=f"Fréquence positive : {prevalence:.2%}")
            ax.set(xlabel="Rappel", ylabel="Précision", title="Precision–Recall OOF agrégée — cinq approches", xlim=(0, 1), ylim=(0, 1.02))
            ax.legend(fontsize=9); ax.grid(alpha=.2)
            fig.text(.5, .02, note + "\nVIX non certifié point-in-time ; courbes sans intervalle de confiance", ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .075, 1, 1)); fig.savefig(output_dir/"precision_recall_curves.png", dpi=160); plt.close(fig)
            for metric, filename, title in [("brier_score", "brier_by_year.png", "Brier annuel — plus bas est préférable"),
                                             ("average_precision", "ap_by_year.png", "Average Precision annuelle — plus haut est préférable")]:
                fig, ax = plt.subplots(figsize=(11, 7))
                for name in APPROACHES:
                    rows = result.metrics.loc[(result.metrics.approach == name) & (result.metrics.period_type == "fold")]
                    ax.plot(rows.validation_year, rows[metric], marker="o", color=colors[name],
                            linestyle="--" if name in BASELINES else "-", label=labels[name])
                ax.set(xlabel="Année de validation", ylabel=metric, title=title, xticks=list(range(2015, 2023)))
                ax.legend(fontsize=9); ax.grid(alpha=.2)
                extra = "\nAP non évaluée en 2017 et 2019 : aucun positif" if metric == "average_precision" else "\nVIX non certifié point-in-time ; aucune calibration effectuée"
                fig.text(.5, .02, note + extra, ha="center", fontsize=9)
                fig.tight_layout(rect=(0, .075, 1, 1)); fig.savefig(output_dir/filename, dpi=160); plt.close(fig)
        finally:
            if previous is None: os.environ.pop("MPLCONFIGDIR", None)
            else: os.environ["MPLCONFIGDIR"] = previous


def export_evaluation(result, output_dir=DEFAULT_EVALUATION):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation directory: {output_dir}")
    if not result.provenance.get("baseline_metrics_verified"):
        raise DataValidationError("Unverified baselines: recommendation/export forbidden")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        result.metrics.to_csv(stage/"model_metrics.csv", index=False)
        (stage/"model_comparison.md").write_text(comparison_markdown(result), encoding="utf-8")
        render_figures(result, stage)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during export: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists(): shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION)
    args = parser.parse_args()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    if branch != "feature/ml1-crisis-detection": parser.exit(2, "Expected feature/ml1-crisis-detection\n")
    if args.output_dir.exists(): parser.exit(2, f"Output exists: {args.output_dir}; choose a new directory\n")
    result = evaluate_saved_predictions()
    export_evaluation(result, args.output_dir)
    print(result.metrics.loc[result.metrics.period_type == "pooled_oof"].to_string(index=False))
    print(json.dumps(result.recommendation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
