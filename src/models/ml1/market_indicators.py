"""Explicit, offline indicator variants. Core remains the reference/default.

The local VIX snapshot is NOT point-in-time certified. Its lagged variant can
only be exported/loaded with allow_experimental=True. Macro integration fails
closed until independent historical availability evidence is validated.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .core import DataValidationError, decision_times, validate_index
from .pipeline import (
    DEFAULT_OUTPUT, DEFAULT_RAW, PROJECT_ROOT, check_calendar, session_dates, sha256,
)
from .time_validation import CONTROL_COLUMNS, FEATURE_COLUMNS

INDICATOR_VERSION = "0.1.0-experimental"
VIX_COLUMNS = ("VIX_level", "VIX_change_5d")
ENRICHED_COLUMNS = FEATURE_COLUMNS + VIX_COLUMNS
DEFAULT_VIX_OUTPUT = PROJECT_ROOT / "data/processed/ml1_core_vix_experimental"
VARIANT_FILES = (
    "features_ml1.parquet", "targets_ml1.parquet", "features_ml1_inference.parquet",
    "target_diagnostics.csv", "vix_session_features.parquet", "vix_lineage.parquet",
    "dataset_metadata.json",
)
CORE_FILES = (
    "features_ml1.parquet", "targets_ml1.parquet", "features_ml1_inference.parquet",
    "target_diagnostics.csv", "dataset_metadata.json",
)
OFFICIAL_REFERENCES = [
    "https://www.cboe.com/tradable-products/vix/vix-historical-data",
    "https://fred.stlouisfed.org/data/VIXCLS",
    "https://fred.stlouisfed.org/docs/api/fred/realtime_period.html",
    "https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html",
]


def variant_profile(variant: str = "core", allow_experimental: bool = False) -> dict:
    if variant == "core":
        return {"name": "core", "status": "reference", "feature_columns": list(FEATURE_COLUMNS),
                "default": True, "inherits_price_receipt_assumptions": True}
    if variant == "core_vix_macro":
        raise DataValidationError("Macro variant blocked: historical availability/vintages not validated")
    if variant != "core_vix":
        raise ValueError(f"Unknown ML1 variant: {variant}")
    if not allow_experimental:
        raise DataValidationError("core_vix is experimental; explicit allow_experimental=True required")
    return {"name": "core_vix", "status": "experimental", "default": False,
            "feature_columns": list(ENRICHED_COLUMNS), "point_in_time_certified": False,
            "requires_experimental_opt_in": True}


def read_vix(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    if not {"observation_date", "VIXCLS"}.issubset(data):
        raise DataValidationError("VIX raw CSV must contain observation_date and VIXCLS")
    dates = session_dates(data.observation_date)
    return pd.Series(pd.to_numeric(data.VIXCLS, errors="coerce").to_numpy(dtype=float),
                     index=dates, name="VIXCLS")


@dataclass(frozen=True)
class VixFeatureSet:
    features: pd.DataFrame
    lineage: pd.DataFrame
    audit: dict


def build_vix_features(
    vix: pd.Series, spy_sessions: pd.DatetimeIndex, lag_sessions: int = 1,
    known_at: pd.Series | None = None,
) -> VixFeatureSet:
    """Lag AFTER alignment to SPY. No fill, row deletion, fitting or downloads.

    Optional, timezone-aware known_at evidence may only REMOVE observations
    unavailable at their decision cutoff. It does not certify source vintages.
    Without evidence, availability is an explicitly experimental assumption.
    Five-session change uses the two historically admissible endpoint levels.
    """
    validate_index(vix.index)
    validate_index(spy_sessions)
    spy_sessions = spy_sessions.copy()
    if type(lag_sessions) is not int or lag_sessions < 1:
        raise ValueError("Unverified VIX requires at least one SPY session of lag")
    numeric = pd.to_numeric(vix, errors="coerce").astype(float)
    clean = numeric.where(np.isfinite(numeric) & (numeric > 0))
    aligned = clean.reindex(spy_sessions)
    level = aligned.shift(lag_sessions)
    source_dates = pd.Series(spy_sessions, index=spy_sessions).shift(lag_sessions)
    evidence = pd.Series(pd.NaT, index=spy_sessions, dtype="datetime64[ns, UTC]")
    rejected = pd.Series(False, index=spy_sessions)
    if known_at is not None:
        validate_index(known_at.index)
        for value in known_at.dropna():
            if pd.Timestamp(value).tzinfo is None:
                raise DataValidationError("Publication evidence must have explicit timezone")
        publication = pd.to_datetime(known_at, utc=True).reindex(spy_sessions)
        evidence = publication.shift(lag_sessions)
        rejected = evidence.isna() | (evidence > decision_times(spy_sessions))
        level = level.mask(rejected)
    features = pd.DataFrame({
        "VIX_level": level,
        "VIX_change_5d": level / level.shift(5) - 1,
    }, index=spy_sessions).replace([np.inf, -np.inf], np.nan)
    features.index.name = "Date"
    lineage = pd.DataFrame({
        "VIX_observation_session": source_dates,
        "VIX_comparison_observation_session": source_dates.shift(5),
        "VIX_known_at": evidence,
        "publication_evidence_supplied": known_at is not None,
        "publication_unavailable_or_future": rejected,
    }, index=spy_sessions)
    lineage.index.name = "Date"
    valid = clean.dropna()
    audit = {
        "series": "VIXCLS", "origin": "Cboe via FRED local snapshot", "unit": "index points",
        "frequency": "daily close", "raw_rows": len(vix),
        "raw_missing": int(numeric.isna().sum()),
        "raw_invalid_nonmissing": int((numeric.notna() & (~np.isfinite(numeric) | (numeric <= 0))).sum()),
        "raw_valid": len(valid), "first_date": str(vix.index.min().date()),
        "last_date": str(vix.index.max().date()), "spy_sessions": len(spy_sessions),
        "missing_on_spy_sessions_before_lag": int(aligned.isna().sum()),
        "valid_non_spy_observations_ignored": len(valid.index.difference(spy_sessions)),
        "lag_spy_sessions": lag_sessions, "imputation": "none; no forward/back fill",
        "known_at_evidence_supplied": known_at is not None,
        "publication_rejected_rows": int(rejected.sum()),
        "missing_by_feature_full_calendar": {c: int(features[c].isna().sum()) for c in VIX_COLUMNS},
        "complete_rows_full_calendar": int(features.notna().all(axis=1).sum()),
        "point_in_time_certified": False,
    }
    return VixFeatureSet(features, lineage, audit)


def _validate_core_metadata(base_dir: Path) -> tuple[dict, dict]:
    metadata = json.loads((base_dir/"dataset_metadata.json").read_text())
    if (tuple(metadata.get("features_retained", [])) != FEATURE_COLUMNS
            or metadata.get("target", {}).get("operational_threshold") != .07
            or metadata.get("target", {}).get("horizon_spy_sessions") != 21
            or metadata.get("temporal_policy", {}).get("decision_time") != "17:00"
            or metadata.get("temporal_policy", {}).get("timezone") != "America/New_York"):
        raise DataValidationError("Expected the preserved core ten-feature dataset contract")
    fingerprints = {name: sha256(base_dir/name) for name in CORE_FILES}
    for name in CORE_FILES[:-1]:
        if metadata.get("output_sha256", {}).get(name) != fingerprints[name]:
            raise DataValidationError(f"Core dataset fingerprint mismatch: {name}")
    return metadata, fingerprints


def build_vix_variant(
    base_dir: Path = DEFAULT_OUTPUT, raw_dir: Path = DEFAULT_RAW,
    output_dir: Path = DEFAULT_VIX_OUTPUT, *, allow_experimental: bool = False,
    lag_sessions: int = 1,
) -> dict:
    profile = variant_profile("core_vix", allow_experimental)
    base_dir, raw_dir, output_dir = (Path(p).resolve() for p in (base_dir, raw_dir, output_dir))
    if (output_dir == base_dir or output_dir.is_relative_to(base_dir) or base_dir.is_relative_to(output_dir)
            or output_dir == raw_dir or output_dir.is_relative_to(raw_dir) or raw_dir.is_relative_to(output_dir)):
        raise DataValidationError("Variant outputs must be separate from core and raw directories")
    if output_dir.is_relative_to(PROJECT_ROOT) and not output_dir.is_relative_to(PROJECT_ROOT/"data/processed"):
        raise DataValidationError("Repository indicator outputs must stay under data/processed/")
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}; choose a new variant run directory")
    base_metadata, core_hashes = _validate_core_metadata(base_dir)
    spy_path, vix_path = raw_dir/"SPY_raw.csv", raw_dir/"VIX_raw.csv"
    spy_source = next(s for s in base_metadata["sources"] if s["file"] == "SPY_raw.csv")
    if sha256(spy_path) != spy_source["sha256"]:
        raise DataValidationError("SPY calendar source changed since core export")
    sessions = session_dates(pd.read_csv(spy_path, usecols=["Date"])["Date"])
    check_calendar(sessions)
    vix = build_vix_features(read_vix(vix_path), sessions, lag_sessions)
    base_features = pd.read_parquet(base_dir/"features_ml1.parquet")
    base_inference = pd.read_parquet(base_dir/"features_ml1_inference.parquet")
    controls = pd.read_parquet(base_dir/"targets_ml1.parquet", columns=list(CONTROL_COLUMNS))
    if (tuple(base_features) != FEATURE_COLUMNS or tuple(base_inference) != FEATURE_COLUMNS
            or not base_features.index.equals(controls.index)):
        raise DataValidationError("Core features and targets must have the expected schema and alignment")
    features = base_features.join(vix.features, how="left")
    inference = base_inference.join(vix.features, how="left")
    if not features.index.equals(base_features.index) or not inference.index.equals(base_inference.index):
        raise DataValidationError("VIX enrichment must never delete core calendar rows")
    pd.testing.assert_frame_equal(features.loc[:, list(FEATURE_COLUMNS)], base_features)
    pd.testing.assert_frame_equal(features, inference.loc[features.index])
    metadata = {
        "indicator_pipeline_version": INDICATOR_VERSION, "profile": profile, "default_variant": "core",
        "base_pipeline_version": base_metadata["pipeline_version"],
        "base_output_sha256": core_hashes, "target": base_metadata["target"],
        "temporal_policy": {**base_metadata["temporal_policy"],
                            "vix_policy": "experimental previous-SPY-session close proxy; not proven PIT"},
        "feature_columns": list(ENRICHED_COLUMNS),
        "vix_source": {"file": vix_path.name, "sha256": sha256(vix_path), **vix.audit},
        "macro_policy": "excluded; historical observation vintages/publication receipt evidence not validated",
        "feature_rules": {"VIX_level": "VIXCLS reindex(SPY_sessions).shift(lag_sessions)",
                          "VIX_change_5d": "admissible_level[t] / admissible_level[t-5_SPY_sessions] - 1",
                          "missing_values": "retain all rows as NA; eligible modelling rows selected explicitly"},
        "quality": {
            "supervised_rows_preserved": len(features), "inference_rows_preserved": len(inference),
            "supervised_complete_rows": int(features.notna().all(axis=1).sum()),
            "inference_complete_rows": int(inference.notna().all(axis=1).sum()),
            "missing_supervised_by_feature": {c:int(features[c].isna().sum()) for c in VIX_COLUMNS},
            "missing_inference_by_feature": {c:int(inference[c].isna().sum()) for c in VIX_COLUMNS},
            "first_supervised_date": str(features.index.min().date()),
            "last_supervised_date": str(features.index.max().date()),
            "latest_inference_date": str(inference.index.max().date()),
        },
        "limitations": [
            "No historical receipt/publication or revision proof in local VIXCLS snapshot.",
            "A one-session lag reduces contemporaneous-use risk; it is not a PIT certificate.",
            "Same-day VIX and all macro variables remain excluded from operational/reference data.",
            "Unknown delays longer than the lag and later source revisions remain possible.",
            "Core ETF price receipt assumptions at 17:00 New York remain unchanged.",
        ],
        "documentation_review_date": "2026-11-14", "official_references": OFFICIAL_REFERENCES,
        "source_code_sha256": {p.name: sha256(p) for p in
                               [Path(__file__), Path(__file__).with_name("indicator_validation.py")]},
        "output_files": list(VARIANT_FILES),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        features.to_parquet(stage/"features_ml1.parquet", engine="pyarrow")
        inference.to_parquet(stage/"features_ml1_inference.parquet", engine="pyarrow")
        vix.features.to_parquet(stage/"vix_session_features.parquet", engine="pyarrow")
        vix.lineage.to_parquet(stage/"vix_lineage.parquet", engine="pyarrow")
        # Opaque copies: do not inspect final-test outcomes or recompute any labels.
        for name in ("targets_ml1.parquet", "target_diagnostics.csv"):
            shutil.copyfile(base_dir/name, stage/name)
        metadata["output_sha256"] = {name:sha256(stage/name) for name in VARIANT_FILES[:-1]}
        (stage/"dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during build: {output_dir}")
        stage.rename(output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["core", "core_vix", "core_vix_macro"], default="core")
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VIX_OUTPUT)
    parser.add_argument("--lag-sessions", type=int, default=1)
    args = parser.parse_args()
    try:
        profile = variant_profile(args.variant, args.allow_experimental)
        if args.variant == "core":
            print(json.dumps(profile, indent=2))
            return
        metadata = build_vix_variant(args.base_dir, args.raw_dir, args.output_dir,
                                     allow_experimental=args.allow_experimental, lag_sessions=args.lag_sessions)
        print(json.dumps({"profile":profile,"vix_audit":metadata["vix_source"],"quality":metadata["quality"]}, indent=2))
    except (DataValidationError, ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"ML1 indicator validation failed: {exc}\n")


if __name__ == "__main__":
    main()
