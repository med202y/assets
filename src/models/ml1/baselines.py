"""Train-only probability baselines for the pre-2023 ML1 development study."""

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .core import DataValidationError, validate_index
from .market_indicators import ENRICHED_COLUMNS
from .time_validation import FEATURE_COLUMNS, FINAL_TEST_START, LABEL

VIX_THRESHOLD = 25.0
SMOOTHING_ALPHA = 1.0
MIN_GROUP_SIZE = 30


def validate_development_features(X: pd.DataFrame, columns: tuple) -> None:
    if not isinstance(X, pd.DataFrame) or tuple(X.columns) != columns:
        raise DataValidationError("Baseline requires the explicit core/core_vix feature schema")
    validate_index(X.index)
    # Reject holdout dates before inspecting any feature/label values.
    if (X.index >= FINAL_TEST_START).any():
        raise DataValidationError("2023+ is a closed final test, outside this baseline study")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise DataValidationError("Baseline features must be finite; no implicit imputation")


def validate_development_labels(y: pd.Series) -> None:
    if not isinstance(y, pd.Series) or y.name != LABEL:
        raise DataValidationError(f"Only the Series {LABEL} may be used as y")
    validate_index(y.index)
    if (y.index >= FINAL_TEST_START).any():
        raise DataValidationError("2023+ label values must never enter baseline evaluation")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise DataValidationError("Development labels must be complete and binary")


def _fit_inputs(X, y, columns):
    validate_development_features(X, columns)
    if not isinstance(y, pd.Series) or not X.index.equals(y.index):
        raise DataValidationError("Training X and y must have identical dates")
    validate_development_labels(y)
    return y.to_numpy(dtype=np.int8)


def _probabilities(positive):
    positive = np.asarray(positive, dtype=float)
    if positive.ndim != 1 or not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise DataValidationError("Predicted probabilities must be finite and within [0, 1]")
    return np.column_stack((1 - positive, positive))


class HistoricalFrequencyBaseline:
    """One unsmoothed positive frequency, recomputed on each fold's y_train."""

    def get_params(self, deep=True):
        return {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        truth = _fit_inputs(X, y, FEATURE_COLUMNS)
        self.n_samples_ = len(truth)
        self.n_positive_ = int(truth.sum())
        self.probability_ = self.n_positive_ / self.n_samples_
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.training_summary_ = {
            "n_train": self.n_samples_, "n_positive_train": self.n_positive_,
            "global_train_frequency": self.probability_, "smoothing": "none",
        }
        return self

    def predict_proba(self, X: pd.DataFrame):
        if not hasattr(self, "probability_"):
            raise NotFittedError("HistoricalFrequencyBaseline must be fitted on y_train first")
        validate_development_features(X, FEATURE_COLUMNS)
        return _probabilities(np.full(len(X), self.probability_))


class VixThresholdBaseline:
    """Experimental lagged-VIX buckets, with Beta(alpha, alpha) smoothing.

    Low: VIX_level < 25. High: VIX_level >= 25 (equality belongs to high).
    Groups with n < min_group_size fall back to the unsmoothed global training
    frequency, including empty groups. Threshold search is intentionally refused.
    Provenance/one-SPY-session lag is enforced by the evaluation adapter.
    """

    def __init__(self, threshold=VIX_THRESHOLD, alpha=SMOOTHING_ALPHA, min_group_size=MIN_GROUP_SIZE):
        if threshold != VIX_THRESHOLD:
            raise ValueError("This experiment fixes the VIX threshold at 25; no threshold search")
        if not isinstance(alpha, (float, int)) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if type(min_group_size) is not int or min_group_size < 1:
            raise ValueError("min_group_size must be a positive integer")
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.min_group_size = min_group_size

    def get_params(self, deep=True):
        return {"threshold": self.threshold, "alpha": self.alpha, "min_group_size": self.min_group_size}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        truth = _fit_inputs(X, y, ENRICHED_COLUMNS)
        self.global_probability_ = float(truth.mean())
        high = X.VIX_level.to_numpy(dtype=float) >= self.threshold
        self.group_probabilities_ = {}
        groups = {}
        for name, mask in [("low", ~high), ("high", high)]:
            n = int(mask.sum())
            positives = int(truth[mask].sum())
            fallback = n < self.min_group_size
            probability = self.global_probability_ if fallback else (positives+self.alpha)/(n+2*self.alpha)
            self.group_probabilities_[name] = float(probability)
            groups[name] = {
                "n_train": n, "n_positive_train": positives, "fallback_to_global": fallback,
                "probability": float(probability),
            }
        self.classes_ = np.array([0, 1], dtype=np.int8)
        self.training_summary_ = {
            "n_train": len(truth), "n_positive_train": int(truth.sum()),
            "global_train_frequency": self.global_probability_,
            "smoothing": "Beta(alpha, alpha) posterior mean for groups with sufficient support",
            "groups": groups,
        }
        return self

    def predict_proba(self, X: pd.DataFrame):
        if not hasattr(self, "group_probabilities_"):
            raise NotFittedError("VixThresholdBaseline must be fitted on y_train first")
        validate_development_features(X, ENRICHED_COLUMNS)
        probability = np.where(
            X.VIX_level.to_numpy(dtype=float) >= self.threshold,
            self.group_probabilities_["high"], self.group_probabilities_["low"],
        )
        return _probabilities(probability)
