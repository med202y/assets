"""Pure ML1 transformations. All rolling windows count SPY sessions."""

from decimal import Decimal

import numpy as np
import pandas as pd

HORIZON = 21
DECISION_TIMEZONE = "America/New_York"
DECISION_HOUR = 17
EXCLUDED_FEATURES = {
    "VIX": "VIXCLS snapshot: no historical publication/receipt timestamps or vintages",
    "VIX_chg_5d": "depends on the excluded VIXCLS series",
    "BAA10Y": "FRED snapshot: historical availability and revisions not established",
    "T10Y2Y": "FRED snapshot: historical availability and revisions not established",
    "DGS3MO": "FRED snapshot: historical availability and revisions not established",
    "DGS2": "FRED snapshot: historical availability and revisions not established",
    "DGS10": "FRED snapshot: historical availability and revisions not established",
}


class DataValidationError(ValueError):
    """Input cannot support an honest session-based dataset."""


def validate_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans:
        raise DataValidationError("Expected a nonempty DatetimeIndex without NaT")
    if index.tz is not None or not index.equals(index.normalize()):
        raise DataValidationError("Session index must contain naive normalized dates")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise DataValidationError("Session dates must be unique and strictly increasing")


def validate_spy(spy: pd.Series) -> None:
    validate_index(spy.index)
    if not np.isfinite(spy.to_numpy(dtype=float)).all() or (spy <= 0).any():
        raise DataValidationError("Invalid SPY adjusted price; prices are never filled or dropped")


def decision_times(index: pd.DatetimeIndex) -> pd.Series:
    """17:00 New York, including early closes; preserve daylight saving time."""
    validate_index(index)
    local = (index + pd.Timedelta(hours=DECISION_HOUR)).tz_localize(DECISION_TIMEZONE)
    return pd.Series(local.tz_convert("UTC"), index=index, name="decision_time")


def build_targets(spy: pd.Series) -> pd.DataFrame:
    """No price imputation. Missing/invalid required prices imply nullable labels.

    Price comparisons use Decimal representations to avoid classifying exact
    threshold equality as a breach because P_min / P_t - 1 rounded down.
    """
    validate_index(spy.index)
    horizon = HORIZON
    valid_price = spy.where(np.isfinite(spy) & (spy > 0))
    future = pd.concat([valid_price.shift(-j) for j in range(1, horizon + 1)], axis=1)
    complete = future.notna().all(axis=1) & valid_price.notna()
    minimum = future.min(axis=1).where(complete)
    dates = pd.Series(spy.index, index=spy.index)
    result = pd.DataFrame(index=spy.index)
    result["spy_adjusted_close"] = valid_price
    result["future_min_price_21"] = minimum
    result["future_min_return_21"] = minimum / valid_price - 1
    result["target_start_session"] = dates.shift(-1)
    result["target_end_session"] = dates.shift(-horizon)
    result["future_session_count"] = future.notna().sum(axis=1).astype("int16")
    for suffix, multiplier in [("07", Decimal("0.93")), ("10", Decimal("0.90"))]:
        labels = [
            int(Decimal(str(low)) < Decimal(str(price)) * multiplier) if good else pd.NA
            for price, low, good in zip(valid_price, minimum, complete, strict=True)
        ]
        result[f"y_crisis_21d_{suffix}"] = pd.array(labels, dtype="Int8")
    result["y_crisis_21d"] = result["y_crisis_21d_07"].copy()
    return result


def build_features(spy: pd.Series, ief: pd.Series | None = None) -> pd.DataFrame:
    """Historical price-only features. No FRED inputs, fitting, or scaling."""
    validate_spy(spy)
    ret = spy.pct_change(fill_method=None)
    features = pd.DataFrame(index=spy.index)
    for window in (1, 5, 21, 63):
        features[f"SPY_ret_{window}d"] = spy.pct_change(window, fill_method=None)
    for window in (5, 21, 63):
        features[f"SPY_vol_{window}d"] = ret.rolling(window).std(ddof=1) * np.sqrt(252)
    features["SPY_drawdown_252d"] = spy / spy.rolling(252).max() - 1
    features["SPY_vs_ma200"] = spy / spy.rolling(200).mean() - 1
    if ief is not None:
        validate_index(ief.index)
        aligned = ief.reindex(spy.index).where(lambda s: np.isfinite(s) & (s > 0))
        ief_ret = aligned.pct_change(fill_method=None)
        features["SPY_IEF_corr_63d"] = ret.rolling(63).corr(ief_ret)
    return features.replace([np.inf, -np.inf], np.nan)


def align_known_observations(observations: pd.DataFrame, decisions: pd.Series) -> pd.DataFrame:
    """Availability gate for future verified sources; unused by default.

    Required columns: observation_date (session date), known_at (aware timestamp),
    value. At each cutoff use the latest observation, and its latest then-known
    revision. With no eligible record, leave NA. Does not assert that a supplied
    timestamp is authentic and performs no imputation or implicit one-day lag.
    """
    required = {"observation_date", "known_at", "value"}
    if not required.issubset(observations):
        raise DataValidationError("Publication evidence required: observation_date, known_at, value")
    validate_index(decisions.index)
    data = observations.copy()
    try:
        data["observation_date"] = pd.to_datetime(data["observation_date"])
        for value in data["known_at"]:
            if pd.isna(value) or pd.Timestamp(value).tzinfo is None:
                raise DataValidationError("known_at must have an explicit timezone")
        data["known_at"] = pd.to_datetime(data["known_at"], utc=True)
        for value in decisions:
            if pd.isna(value) or pd.Timestamp(value).tzinfo is None:
                raise DataValidationError("Decision cutoffs must have an explicit timezone")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(str(exc)) from exc
    if data["observation_date"].isna().any():
        raise DataValidationError("Missing observation date")
    if data.duplicated(["observation_date", "known_at"]).any():
        raise DataValidationError("Ambiguous versions with the same availability timestamp")
    data = data.sort_values(["observation_date", "known_at"])
    rows = []
    for date, cutoff in decisions.items():
        eligible = data.loc[(data["known_at"] <= cutoff) & (data["observation_date"] <= date)]
        if eligible.empty:
            rows.append({"value": np.nan, "known_at": pd.NaT, "observation_date": pd.NaT})
        else:
            rows.append(eligible.iloc[-1][["value", "known_at", "observation_date"]].to_dict())
    return pd.DataFrame(rows, index=decisions.index)


def target_diagnostics(targets: pd.DataFrame) -> pd.DataFrame:
    """Only training 2005–2014, with each entire target window inside 2014."""
    train = targets.loc[
        (targets.index >= "2005-01-01")
        & (targets.index <= "2014-12-31")
        & (targets["target_end_session"] <= pd.Timestamp("2014-12-31"))
    ]
    rows = []
    groups = [("training_total", train)] + [
        (str(year), train.loc[train.index.year == year]) for year in range(2005, 2015)
    ]
    for period, group in groups:
        for suffix, threshold in [("07", 0.07), ("10", 0.10)]:
            sample = group.loc[group[f"y_crisis_21d_{suffix}"].notna()]
            positives = int(sample[f"y_crisis_21d_{suffix}"].sum())
            rows.append({
                "period": period, "threshold": threshold, "horizon_sessions": HORIZON,
                "n_observations": len(sample), "positive_labels": positives,
                "negative_labels": len(sample) - positives,
                "positive_frequency": positives / len(sample) if len(sample) else None,
                "first_decision_session": str(sample.index.min().date()) if len(sample) else None,
                "last_decision_session": str(sample.index.max().date()) if len(sample) else None,
                "last_target_end": str(sample["target_end_session"].max().date()) if len(sample) else None,
                "all_horizons_21": bool((sample["future_session_count"] == HORIZON).all()),
                "scope": "raw_SPY_calendar_only; overlapping labels; not independent crises",
            })
    return pd.DataFrame(rows)
