"""The ONLY bridge between the Streamlit UI (``app/``) and the backend (``src/``).

Rules enforced here:

* pages and components never import ``src``;
* nothing is ever retrained or re-optimised on a widget interaction -- model
  outputs are read from precomputed artifacts and expensive calls are cached;
* missing artifacts/dependencies degrade to clearly-labelled deterministic
  placeholder values instead of raising a traceback in the UI;
* the FRED API key is never read, printed or displayed by the frontend.

Public functions used by the pages:

``get_data_as_of``, ``get_date_bounds``, ``get_market_overview``,
``get_ml1_signal``, ``get_ml2_vol``, ``get_allocation``,
``get_transaction_cost``, ``get_backtest_results``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from app import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "src" / "data" / "financial_dataset" / "processed"
ML2_DIR = PROJECT_ROOT / "src" / "models" / "ml2"
ML1_TRAINING = PROJECT_ROOT / "data" / "processed" / "ml1_training"
ML1_RELEASE = PROJECT_ROOT / "data" / "processed" / "ml1_inference_release"
ML1_REPORTS = PROJECT_ROOT / "reports" / "ml1"

# Backtest assumption: 10 bps per unit of turnover. There is no src/backtest
# implementation yet, so this is the documented value used by the placeholder.
TRANSACTION_COST = 0.001

RISK_AVERSION = {"conservative": 3.0, "balanced": 1.5, "aggressive": 0.6}
PLACEHOLDER_WEIGHTS = {
    "conservative": {"SPY": 0.20, "IEF": 0.45, "LQD": 0.25, "VNQ": 0.10},
    "balanced": {"SPY": 0.35, "IEF": 0.30, "LQD": 0.20, "VNQ": 0.15},
    "aggressive": {"SPY": 0.50, "IEF": 0.15, "LQD": 0.15, "VNQ": 0.20},
}
BENCHMARK_WEIGHTS = {
    "SPY_ONLY": {"SPY": 1.0, "IEF": 0.0, "LQD": 0.0, "VNQ": 0.0},
    "6040": {"SPY": 0.6, "IEF": 0.4, "LQD": 0.0, "VNQ": 0.0},
    "EQW": {asset: 0.25 for asset in config.ASSETS},
}

SOURCE_BACKEND = "backend"
SOURCE_PLACEHOLDER = "placeholder"


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _exists(path: Path) -> bool:
    return path.is_file()


def _synthetic_prices() -> pd.DataFrame:
    """Deterministic last-resort prices when the data cache is absent."""
    index = pd.bdate_range(end=date(2026, 11, 4), periods=1800)
    rng = np.random.default_rng(11)
    drift = np.array([0.08, 0.035, 0.045, 0.06]) / 252
    vol = np.array([0.16, 0.06, 0.08, 0.19]) / math.sqrt(252)
    corr = np.array(
        [
            [1.0, -0.15, 0.25, 0.60],
            [-0.15, 1.0, 0.55, -0.05],
            [0.25, 0.55, 1.0, 0.20],
            [0.60, -0.05, 0.20, 1.0],
        ]
    )
    cov = np.outer(vol, vol) * corr
    shocks = rng.multivariate_normal(drift, cov, len(index))
    values = 100 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(values, index=index, columns=list(config.ASSETS))


@st.cache_data(show_spinner=False)
def _prices() -> pd.DataFrame:
    """Read the cleaned price cache produced by ``src/data``."""
    csv = PROCESSED / "final_dataset.csv"
    try:
        if not _exists(csv):
            raise FileNotFoundError(f"missing {csv}")
        frame = pd.read_csv(csv, index_col=0, parse_dates=True)
        frame = frame.loc[:, list(config.ASSETS)].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna().sort_index()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        if frame.empty:
            raise ValueError("empty price cache")
        return frame
    except Exception as exc:  # noqa: BLE001
        logger.warning("Using synthetic prices: %s", exc)
        return _synthetic_prices()


@st.cache_data(show_spinner=False)
def _features_ml2() -> pd.DataFrame:
    path = PROCESSED / "features_ml2.parquet"
    if not _exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path).sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read ML2 features: %s", exc)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _features_ml1() -> pd.DataFrame:
    path = PROCESSED / "features_ml1.parquet"
    if not _exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path).sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read ML1 features: %s", exc)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _ml1_series() -> tuple[pd.DataFrame, str]:
    """Precomputed ML 1 probabilities (OOF + blind holdout), never refitted."""
    pieces: list[pd.Series] = []
    oof_path = ML1_TRAINING / "ml_models_oof_predictions.parquet"
    try:
        if _exists(oof_path):
            oof = pd.read_parquet(oof_path)
            oof = oof.loc[oof["model"].eq("logistic_regression")]
            series = pd.Series(
                oof["predicted_probability"].to_numpy(dtype=float),
                index=pd.to_datetime(oof["prediction_date"]).dt.normalize(),
                name="probability",
            )
            pieces.append(series)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read ML 1 OOF probabilities: %s", exc)

    blind_path = ML1_RELEASE / "monthly_blind_probabilities.csv"
    try:
        if _exists(blind_path):
            blind = pd.read_csv(blind_path)
            series = pd.Series(
                blind["p_crisis"].to_numpy(dtype=float),
                index=pd.to_datetime(blind["prediction_date"]).dt.normalize(),
                name="probability",
            )
            pieces.append(series)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read ML 1 blind probabilities: %s", exc)

    if pieces:
        history = pd.concat(pieces).sort_index()
        history = history[~history.index.duplicated(keep="last")].to_frame()
        return history, SOURCE_BACKEND

    # Deterministic placeholder derived from real ML 1 features.
    features = _features_ml1()
    if features.empty:
        return pd.DataFrame(columns=["probability"]), SOURCE_PLACEHOLDER
    score = (
        -3.2
        + 7.0 * features["SPY_drawdown_252d"].abs().clip(upper=0.3)
        + 9.0 * features["SPY_vol_21d"].clip(upper=0.4)
        + 1.2 * (features.get("VIX", pd.Series(20.0, index=features.index)) / 100.0)
    )
    history = (1.0 / (1.0 + np.exp(-score))).clip(0.01, 0.85).to_frame("probability")
    return history, SOURCE_PLACEHOLDER


def _latest_ml1_probabilities() -> tuple[float, float, str]:
    history, source = _ml1_series()
    if history.empty:
        return 0.15, 0.12, SOURCE_PLACEHOLDER
    values = history["probability"].dropna()
    current = float(values.iloc[-1])
    previous = float(values.iloc[-2]) if len(values) > 1 else current
    return current, previous, source


@st.cache_data(show_spinner=False)
def _signal_date() -> pd.Timestamp:
    features = _features_ml2()
    if not features.empty:
        return pd.Timestamp(features.index[-1])
    return pd.Timestamp(_prices().index[-1])


@st.cache_data(show_spinner=False)
def _ml2_forecasts(date_value: str) -> tuple[pd.Series, str]:
    """Current ML 2 volatility forecast (precomputed OOS predictions)."""
    stamp = pd.Timestamp(date_value)
    try:
        from src.models.ml2.predict_volatility import predict_volatility

        vols, _cov = predict_volatility(stamp, use_oos=True)
        return pd.Series(vols, dtype=float).reindex(list(config.ASSETS)), SOURCE_BACKEND
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML 2 forecast unavailable, using trailing volatility: %s", exc)

    features = _features_ml2()
    if not features.empty and stamp in features.index:
        row = features.loc[stamp]
        return pd.Series(
            {a: float(row.get(f"{a}_vol_21d", np.nan)) for a in config.ASSETS},
            dtype=float,
        ), SOURCE_PLACEHOLDER
    returns = _prices().pct_change().dropna().iloc[-252:]
    vols = returns[list(config.ASSETS)].std() * math.sqrt(252)
    return vols.reindex(list(config.ASSETS)), SOURCE_PLACEHOLDER


@st.cache_data(show_spinner=False)
def _covariance(date_value: str) -> tuple[pd.DataFrame, str]:
    stamp = pd.Timestamp(date_value)
    try:
        from src.models.ml2.predict_volatility import predict_volatility

        _vols, cov = predict_volatility(stamp, use_oos=True)
        return pd.DataFrame(cov).reindex(index=list(config.ASSETS), columns=list(config.ASSETS)), SOURCE_BACKEND
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML 2 covariance unavailable, using realised covariance: %s", exc)
        returns = _prices().pct_change().dropna().iloc[-252:]
        cov = returns[list(config.ASSETS)].cov() * 252
        return cov, SOURCE_PLACEHOLDER


def _expected_returns() -> np.ndarray:
    """Historical, shrunk expected returns from the backend (past data only)."""
    try:
        from src.optimizer.portfolio_optimizer import historical_expected_returns

        returns = _prices().pct_change().dropna().iloc[-2520:]
        return historical_expected_returns(returns, shrink=0.5)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Expected returns unavailable: %s", exc)
        return np.array([0.07, 0.03, 0.04, 0.05])


def _profile_with_bond_floor(profile: str, min_bonds: float, probability: float) -> Any:
    """Translate the UI's bond-share floor into the backend's equity cap.

    The four assets are exactly bonds (IEF, LQD) plus equity (SPY, VNQ), so
    ``equity <= 1 - bond_floor`` is equivalent to ``bonds >= bond_floor``.
    """
    from src.optimizer.profiles import get_profile

    base = get_profile(profile)
    bond_floor = float(min_bonds) / 100.0
    p = min(max(float(probability), 0.0), 0.95)
    max_equity = min(base.max_equity, (1.0 - bond_floor) / max(1.0 - p, 0.05))
    return replace(base, max_equity=max(0.0, max_equity))


def _solve_weights(
    profile: str,
    min_bonds: float,
    probability: float,
    sigma: pd.DataFrame,
    mu: np.ndarray,
    max_weight: float,
) -> tuple[dict[str, float], str, list[str]]:
    """Solve with the real CVXPY backend, or use a labelled placeholder."""
    try:
        from src.optimizer.recommender import recommend

        rec = recommend(
            capital=100_000.0,
            profile=_profile_with_bond_floor(profile, min_bonds, probability),
            sigma=sigma,
            crisis_probability=float(probability),
            expected_returns=mu,
            risk_aversion=RISK_AVERSION.get(profile, 1.5),
            max_weight=float(max_weight) / 100.0,
        )
        weights = {a: float(rec.weights[a]) * 100.0 for a in config.ASSETS}
        return weights, rec.status, list(rec.warnings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Optimizer unavailable, using placeholder weights: %s", exc)
        return _placeholder_weights(profile, probability, max_weight, min_bonds), SOURCE_PLACEHOLDER, []


def _placeholder_weights(
    profile: str,
    probability: float,
    max_weight: float,
    min_bonds: float,
) -> dict[str, float]:
    """Deterministic, constraint-respecting suggestion when CVXPY is absent."""
    base = dict(PLACEHOLDER_WEIGHTS.get(profile, PLACEHOLDER_WEIGHTS["balanced"]))
    # Equities are trimmed as the crash probability rises.
    trim = min(0.5, max(0.0, probability - 0.10))
    base["SPY"] *= 1.0 - trim
    base["VNQ"] *= 1.0 - 0.5 * trim
    cap = max_weight / 100.0
    for asset in base:
        base[asset] = min(base[asset], cap)
    bond_share = base["IEF"] + base["LQD"]
    floor = min_bonds / 100.0
    if bond_share < floor:
        need = floor - bond_share
        spy, vnq = base["SPY"], base["VNQ"]
        equity = min(spy + vnq, need)
        total_equity = max(spy + vnq, 1e-9)
        base["SPY"] = spy - equity * spy / total_equity
        base["VNQ"] = vnq - equity * vnq / total_equity
        base["IEF"] += need / 2
        base["LQD"] += need / 2
    total = sum(base.values())
    return {a: float(base[a] / total * 100.0) for a in config.ASSETS}


def _benchmark_weights(benchmark: str) -> dict[str, float]:
    return {a: BENCHMARK_WEIGHTS.get(benchmark, BENCHMARK_WEIGHTS["6040"])[a] * 100.0 for a in config.ASSETS}


# --------------------------------------------------------------------------- #
# Public: data / market overview
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_data_as_of() -> str:
    return pd.Timestamp(_prices().index[-1]).strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def get_date_bounds() -> dict[str, date]:
    index = _prices().index
    last = pd.Timestamp(index[-1])
    first = pd.Timestamp(index[0]).date()
    default_start = max(first, (last - pd.DateOffset(years=3)).date())
    return {"min_date": first, "max_date": last.date(), "default_start": default_start}


@st.cache_data(show_spinner=False)
def get_prices(start: date, end: date) -> pd.DataFrame:
    prices = _prices()
    mask = (prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))
    return prices.loc[mask]


@st.cache_data(show_spinner=False)
def get_market_overview(start: date, end: date) -> dict[str, pd.DataFrame]:
    prices = get_prices(start, end)
    if prices.empty:
        empty = pd.DataFrame()
        return {"prices_normalized": empty, "correlation": empty, "drawdowns": empty}
    normalized = prices / prices.iloc[0] * 100.0
    returns = prices.pct_change().dropna()
    correlation = returns.corr() if not returns.empty else pd.DataFrame()
    drawdowns = (prices / prices.cummax() - 1.0) * 100.0
    return {
        "prices_normalized": normalized,
        "correlation": correlation,
        "drawdowns": drawdowns,
    }


# --------------------------------------------------------------------------- #
# Public: risk signals
# --------------------------------------------------------------------------- #

def _large_drawdown_dates(prices: pd.DataFrame, threshold: float = -0.15) -> list[pd.Timestamp]:
    spy = prices["SPY"].dropna()
    if spy.empty:
        return []
    drawdown = spy / spy.cummax() - 1.0
    under = drawdown <= threshold
    episodes = under & ~under.shift(1, fill_value=False)
    dates = [pd.Timestamp(d) for d in spy.index[episodes]][-12:]
    return dates


@st.cache_data(show_spinner=False)
def get_ml1_signal() -> dict[str, Any]:
    history, source = _ml1_series()
    if history.empty:
        history = pd.DataFrame(columns=["probability"])
        current = 0.15
    else:
        current = float(history["probability"].dropna().iloc[-1])

    # Pooled out-of-sample metrics saved by src/models/ml1/evaluate_ml1.py.
    quality = pd.DataFrame(
        [
            {"metric": "auc", "value": 0.545},
            {"metric": "brier", "value": 0.0995},
        ]
    )
    metrics_path = ML1_REPORTS / "evaluation" / "model_metrics.csv"
    try:
        if _exists(metrics_path):
            saved = pd.read_csv(metrics_path)
            row = saved.loc[
                saved["approach"].eq("logistic_regression")
                & saved["period_type"].eq("pooled_oof")
            ].iloc[0]
            quality.loc[quality["metric"].eq("auc"), "value"] = float(row["roc_auc"])
            quality.loc[quality["metric"].eq("brier"), "value"] = float(row["brier_score"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Using default ML 1 metrics: %s", exc)

    auc = float(quality.loc[quality["metric"].eq("auc"), "value"].iloc[0])

    features = pd.DataFrame(columns=["feature", "importance"])
    shap_path = ML1_REPORTS / "calibration" / "shap_feature_importance.csv"
    try:
        if _exists(shap_path):
            shap = pd.read_csv(shap_path).head(8)
            features = shap[["feature", "mean_abs_shap_log_odds"]].rename(
                columns={"mean_abs_shap_log_odds": "importance"}
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read ML 1 feature importance: %s", exc)

    return {
        "current_probability": current,
        "history": history,
        "large_drawdown_dates": _large_drawdown_dates(_prices()),
        "top_features": features,
        "quality_table": quality,
        "weak": bool(auc < 0.60),
        "source": source,
    }


@st.cache_data(show_spinner=False)
def get_ml2_vol() -> dict[str, Any]:
    stamp = _signal_date()
    vols, source = _ml2_forecasts(stamp.strftime("%Y-%m-%d"))
    features = _features_ml2()

    long_run: dict[str, float] = {}
    for asset in config.ASSETS:
        column = f"{asset}_vol_21d"
        if not features.empty and column in features:
            long_run[asset] = float(features[column].dropna().mean())
        else:
            long_run[asset] = float(vols.get(asset, 0.15))

    forecast_table = pd.DataFrame(
        {
            "forecast": [float(vols.get(a, np.nan)) for a in config.ASSETS],
            "long_run": [long_run[a] for a in config.ASSETS],
        },
        index=list(config.ASSETS),
    )

    quality = pd.DataFrame([{"metric": "mae", "value": 0.04}])
    metrics_path = ML2_DIR / "model_metrics.csv"
    try:
        if _exists(metrics_path):
            saved = pd.read_csv(metrics_path)
            test = saved.loc[saved["periode"].eq("test")]
            if not test.empty:
                best = test.groupby("modele")["QLIKE"].mean().idxmin()
                quality.loc[0, "value"] = float(test.loc[test["modele"].eq(best), "MAE"].mean())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Using default ML 2 metric: %s", exc)

    features_table = pd.DataFrame(columns=["feature", "importance"])
    try:
        import joblib

        bundle = joblib.load(ML2_DIR / "best_model.pkl")
        names = list(bundle.get("features", []))
        coefs = []
        for asset in config.ASSETS:
            estimator = bundle["models"][asset]
            model = getattr(estimator, "model", estimator)
            coefs.append(np.asarray(getattr(model, "coef_", np.zeros(len(names))), dtype=float))
        averaged = np.mean(np.abs(np.array(coefs)), axis=0)
        order = np.argsort(-averaged)[:8]
        features_table = pd.DataFrame(
            {
                "feature": [names[i] for i in order],
                "importance": [float(averaged[i]) for i in order],
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read ML 2 feature importance: %s", exc)

    return {
        "forecast_table": forecast_table,
        "quality_table": quality,
        "top_features": features_table,
        "source": source,
    }


# --------------------------------------------------------------------------- #
# Public: allocation
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_allocation(
    profile: str,
    max_weight: float,
    min_bonds: float,
    benchmark: str,
    language: str = config.DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Call the real optimizer with precomputed ML signals; never retrain."""
    current_p, previous_p, probability_source = _latest_ml1_probabilities()
    stamp = _signal_date().strftime("%Y-%m-%d")
    sigma, sigma_source = _covariance(stamp)
    mu = _expected_returns()
    source = (
        SOURCE_BACKEND
        if probability_source == SOURCE_BACKEND and sigma_source == SOURCE_BACKEND
        else SOURCE_PLACEHOLDER
    )

    # Cheap, explicit feasibility checks produce a clear message instead of a
    # solver failure or a silently relaxed constraint.
    if float(max_weight) * len(config.ASSETS) < 100.0 - 1e-9:
        return {
            "status": "infeasible",
            "message": config.label("infeasible_hint", language),
        }
    if float(min_bonds) > 2.0 * float(max_weight) + 1e-9:
        return {
            "status": "infeasible",
            "message": config.label("infeasible_hint", language),
        }

    weights, status, warnings = _solve_weights(
        profile, min_bonds, current_p, sigma, mu, max_weight
    )
    previous_weights, _prev_status, _ = _solve_weights(
        profile, min_bonds, previous_p, sigma, mu, max_weight
    )

    if status not in ("optimal", SOURCE_PLACEHOLDER):
        return {
            "status": "infeasible",
            "message": " ".join(warnings) or config.label("infeasible_hint", language),
        }

    comparison = pd.DataFrame(
        {
            "current": pd.Series(weights),
            "previous": pd.Series(previous_weights),
            f"benchmark_{benchmark}": pd.Series(_benchmark_weights(benchmark)),
        }
    )

    spy_weight = weights["SPY"]
    bond_share = weights["IEF"] + weights["LQD"]
    if current_p >= 0.15:
        spy_line = config.label(
            "explain_spy_reduced",
            language,
            weight=spy_weight,
            p=current_p,
            vol=float(_ml2_forecasts(stamp)[0].get("SPY", 0.0)),
        )
    else:
        spy_line = config.label(
            "explain_spy_kept", language, weight=spy_weight, p=current_p
        )
    lines = [
        spy_line,
        config.label(
            "explain_bonds", language, bonds=bond_share, floor=float(min_bonds)
        ),
        config.label(
            "explain_profile",
            language,
            profile=config.label(f"profile_{profile}", language),
        ),
    ]
    if status == SOURCE_PLACEHOLDER:
        lines.append(config.label("placeholder_notice", language))

    return {
        "status": "ok",
        "weights": weights,
        "comparison": comparison,
        "explanation": "\n\n".join(lines),
        "source": source,
        "portfolio_vol": None,
        # Annual returns (decimals) used by the optimizer, exposed so the UI can
        # show an explicitly labelled estimate without recomputing anything.
        "expected_returns": {
            asset: float(mu[index]) for index, asset in enumerate(config.ASSETS)
        },
        "as_of": get_data_as_of(),
    }


# --------------------------------------------------------------------------- #
# Public: backtest stub (src/backtest is empty in this repository)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_transaction_cost() -> float:
    return TRANSACTION_COST


def _benchmark_returns(
    returns: pd.DataFrame, benchmarks: Sequence[str]
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for benchmark in benchmarks:
        weights = BENCHMARK_WEIGHTS.get(benchmark, BENCHMARK_WEIGHTS["6040"])
        series[benchmark] = sum(
            returns[asset] * share for asset, share in weights.items()
        )
    return pd.DataFrame(series)


@st.cache_data(show_spinner=False)
def get_backtest_results(benchmarks: tuple[str, ...]) -> dict[str, Any]:
    """Placeholder simulation on real prices until ``src/backtest`` exists."""
    prices = _prices().iloc[-2520:]
    returns = prices.pct_change().dropna()
    if returns.empty:
        empty = pd.DataFrame()
        return {
            "equity_curve": empty,
            "metrics": empty,
            "weights_history": empty,
            "underperforms": False,
            "source": SOURCE_PLACEHOLDER,
        }

    target = np.array([0.40, 0.30, 0.20, 0.10])
    # Rebalance on the first trading day of each new month.
    periods = returns.index.to_period("M")
    rebalance_positions = {i for i in range(1, len(returns)) if periods[i] != periods[i - 1]}

    current = target.copy()
    strategy_returns = pd.Series(0.0, index=returns.index)
    turnover_events: dict[pd.Timestamp, float] = {}
    weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
    for i, timestamp in enumerate(returns.index):
        if i in rebalance_positions:
            traded = float(np.abs(target - current).sum())
            turnover_events[timestamp] = traded
            current = target.copy()
            weight_rows[timestamp] = {a: float(w * 100) for a, w in zip(config.ASSETS, current)}
        row = returns.iloc[i][list(config.ASSETS)].to_numpy(dtype=float)
        strategy_returns.iloc[i] = float(current @ row) - TRANSACTION_COST * turnover_events.get(timestamp, 0.0)

    curves = pd.DataFrame({"strategy": (1.0 + strategy_returns).cumprod() * 100.0})
    benchmark_frame = _benchmark_returns(returns, benchmarks)
    for column in benchmark_frame.columns:
        curves[column] = (1.0 + benchmark_frame[column]).cumprod() * 100.0

    def metrics(series: pd.Series) -> dict[str, float]:
        n = len(series)
        total = float(series.iloc[-1] / series.iloc[0])
        annual = total ** (252.0 / max(n, 1)) - 1.0
        daily = series.pct_change().dropna()
        volatility = float(daily.std() * math.sqrt(252))
        sharpe = annual / volatility if volatility > 0 else 0.0
        max_dd = float((series / series.cummax() - 1.0).min())
        return {
            "annual_return": annual * 100,
            "volatility": volatility * 100,
            "sharpe": sharpe,
            "max_drawdown": max_dd * 100,
        }

    metric_rows: dict[str, dict[str, float]] = {"strategy": metrics(curves["strategy"])}
    for column in benchmark_frame.columns:
        metric_rows[column] = metrics(curves[column])
    turnover_per_year = (
        sum(turnover_events.values()) / max(len(returns) / 252.0, 1.0) * 100.0
    )
    metric_rows["strategy"]["turnover"] = turnover_per_year
    for column in benchmark_frame.columns:
        metric_rows[column]["turnover"] = 0.0

    metrics_frame = pd.DataFrame(metric_rows).T
    metrics_frame.index.name = "metric"
    metrics_frame = metrics_frame.reset_index()

    weights_history = pd.DataFrame.from_dict(weight_rows, orient="index")
    weights_history = weights_history.reindex(columns=list(config.ASSETS)).fillna(0.0)

    finals = {c: float(curves[c].iloc[-1]) for c in curves.columns}
    underperforms = bool(benchmarks) and not all(
        finals["strategy"] > finals[b] for b in benchmarks
    )

    return {
        "equity_curve": curves,
        "metrics": metrics_frame,
        "weights_history": weights_history,
        "underperforms": underperforms,
        "source": SOURCE_PLACEHOLDER,
        "transaction_cost": TRANSACTION_COST,
    }
