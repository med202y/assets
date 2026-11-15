"""Explicit variant adapter reusing the unchanged step-3 chronological engine."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import DataValidationError, validate_index
from .market_indicators import DEFAULT_VIX_OUTPUT, ENRICHED_COLUMNS, variant_profile
from .pipeline import DEFAULT_OUTPUT, DEFAULT_RAW, sha256
from .time_validation import (
    FEATURE_COLUMNS, DevelopmentData, WalkForwardConfig, WalkForwardFold,
    assert_fold_is_safe, build_walk_forward_folds, load_development_data,
)


@dataclass(frozen=True)
class FeatureVariantData:
    core: DevelopmentData
    X: pd.DataFrame
    profile: dict
    excluded_incomplete_feature_dates: pd.DatetimeIndex


def load_feature_variant(
    variant: str = "core", *, allow_experimental: bool = False,
    base_dir: Path = DEFAULT_OUTPUT, variant_dir: Path = DEFAULT_VIX_OUTPUT,
    raw_dir: Path = DEFAULT_RAW,
) -> FeatureVariantData:
    profile = variant_profile(variant, allow_experimental)
    base_dir, variant_dir, raw_dir = map(Path, (base_dir, variant_dir, raw_dir))
    base = load_development_data(base_dir, raw_dir/"SPY_raw.csv")
    if variant == "core":
        return FeatureVariantData(base, base.X.copy(), profile, pd.DatetimeIndex([]))
    metadata = json.loads((variant_dir/"dataset_metadata.json").read_text())
    if (metadata.get("profile") != profile or tuple(metadata.get("feature_columns", [])) != ENRICHED_COLUMNS
            or metadata.get("default_variant") != "core"
            or metadata.get("vix_source", {}).get("point_in_time_certified") is not False
            or metadata.get("vix_source", {}).get("lag_spy_sessions", 0) < 1):
        raise DataValidationError("Variant status/schema/temporal convention differs from experimental VIX contract")
    for name, digest in metadata["base_output_sha256"].items():
        if sha256(base_dir/name) != digest:
            raise DataValidationError(f"Variant refers to a different core dataset: {name}")
    for name, digest in metadata["output_sha256"].items():
        if sha256(variant_dir/name) != digest:
            raise DataValidationError(f"Variant fingerprint mismatch: {name}")
    if sha256(raw_dir/"VIX_raw.csv") != metadata["vix_source"]["sha256"]:
        raise DataValidationError("VIX snapshot differs from the experimental variant source")
    # Read only features from the variant. y is loaded by the existing protected core loader.
    all_X = pd.read_parquet(variant_dir/"features_ml1.parquet")
    validate_index(all_X.index)
    if tuple(all_X.columns) != ENRICHED_COLUMNS or not base.X.index.isin(all_X.index).all():
        raise DataValidationError("Variant features lack the expected schema/development dates")
    X = all_X.loc[base.X.index].copy()
    pd.testing.assert_frame_equal(X.loc[:,list(FEATURE_COLUMNS)], base.X)
    complete = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    excluded = X.index[~complete]
    X = X.loc[complete].copy()
    if X.empty:
        raise DataValidationError("No complete experimental development observations")
    core = replace(base, X=base.X.loc[X.index].copy(), y=base.y.loc[X.index].copy(),
                   target_info=base.target_info.loc[X.index].copy())
    return FeatureVariantData(core, X, {**profile, "lag_spy_sessions":metadata["vix_source"]["lag_spy_sessions"]}, excluded)


def build_feature_variant_folds(data: FeatureVariantData, config: WalkForwardConfig | None = None):
    """Indices refer to this variant's complete development rows; no post-2023 y."""
    _validate_variant_schema(data)
    return build_walk_forward_folds(data.core, config)


def _validate_variant_schema(data: FeatureVariantData) -> None:
    expected = FEATURE_COLUMNS if data.profile["name"] == "core" else ENRICHED_COLUMNS
    if data.profile["name"] not in {"core", "core_vix"} or tuple(data.X.columns) != expected:
        raise DataValidationError("Invalid variant feature schema")
    if data.profile["name"] == "core_vix" and (
        data.profile.get("status") != "experimental" or data.profile.get("point_in_time_certified") is not False
        or data.profile.get("requires_experimental_opt_in") is not True or data.profile.get("default") is not False
    ):
        raise DataValidationError("Experimental VIX must not be represented as operational or PIT-certified")
    if not data.X.index.equals(data.core.X.index):
        raise DataValidationError("Variant and core projection have different row indices")
    pd.testing.assert_frame_equal(data.X.loc[:,list(FEATURE_COLUMNS)], data.core.X)


def get_feature_variant_fold_xy(data: FeatureVariantData, fold: WalkForwardFold):
    _validate_variant_schema(data)
    assert_fold_is_safe(data.core, fold)
    return (
        data.X.iloc[fold.train_indices].copy(), data.core.y.iloc[fold.train_indices].copy(),
        data.X.iloc[fold.validation_indices].copy(), data.core.y.iloc[fold.validation_indices].copy(),
    )
