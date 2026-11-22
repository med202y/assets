"""Point d'entrée : recommend(...) -> allocation, montants, volatilité, explication."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constraints import effective_equity_cap
from .portfolio_optimizer import (
    DEFAULT_TURNOVER_PENALTY, optimize_portfolio, prepare_sigma,
)
from .profiles import ASSETS, EQUITY_ASSETS, Profile, get_profile


@dataclass
class Recommendation:
    profile: str
    capital: float
    weights: pd.Series
    amounts: pd.Series
    portfolio_vol: float
    equity_cap_profile: float
    equity_cap_effective: float
    crisis_probability: float
    turnover: float | None
    status: str
    warnings: list[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "profile": self.profile, "capital": self.capital,
            "weights": self.weights.to_dict(), "amounts": self.amounts.to_dict(),
            "portfolio_vol": self.portfolio_vol,
            "equity_cap_effective": self.equity_cap_effective,
            "crisis_probability": self.crisis_probability,
            "turnover": self.turnover, "status": self.status,
            "warnings": self.warnings, "explanation": self.explanation,
        }


def _to_cents(weights: np.ndarray, capital: float) -> np.ndarray:
    """Montants arrondis au centime, somme == capital (méthode du plus grand reste)."""
    total = int(round(capital * 100))
    raw = weights * total
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return base / 100.0


def _explain(rec: "Recommendation", profile: Profile) -> str:
    w = rec.weights
    eq = float(w[list(EQUITY_ASSETS)].sum())
    lines = [
        f"Profil {profile.label} : volatilité max {profile.max_vol:.0%}, "
        f"actions max {profile.max_equity:.0%}, obligations d'État min {profile.min_gov_bonds:.0%}.",
        f"Probabilité de crise (ML 1) = {rec.crisis_probability:.0%} : plafond actions ramené de "
        f"{rec.equity_cap_profile:.1%} à {rec.equity_cap_effective:.1%}.",
        "Allocation : " + ", ".join(f"{a} {w[a]:.1%}" for a in ASSETS) + f" (actions {eq:.1%}).",
        f"Volatilité estimée du portefeuille : {rec.portfolio_vol:.1%} (limite {profile.max_vol:.0%}).",
    ]
    if rec.turnover is not None:
        lines.append(f"Rotation vs allocation précédente : {rec.turnover:.1%}.")
    if rec.status != "optimal":
        lines.append(f"Attention, statut = {rec.status} : " + " ".join(rec.warnings))
    return "\n".join(lines)


def recommend(
    capital: float,
    profile: str | Profile,
    sigma,
    crisis_probability: float = 0.0,
    *,
    prev_weights=None,
    alpha: float = 1.0,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    expected_returns=None,
    risk_aversion: float = 1.0,
    max_weight: float | None = None,
    max_turnover: float | None = None,
) -> Recommendation:
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital doit être > 0")
    prof = get_profile(profile)
    cap_eff = effective_equity_cap(prof.max_equity, crisis_probability, alpha)

    out = optimize_portfolio(
        sigma, prof, cap_eff, prev_weights=prev_weights,
        turnover_penalty=turnover_penalty, expected_returns=expected_returns,
        risk_aversion=risk_aversion, max_weight=max_weight, max_turnover=max_turnover,
    )
    S, _, _ = prepare_sigma(sigma)
    w = out.weights
    vol = float(np.sqrt(max(w @ S @ w, 0.0)))

    turnover = None
    if prev_weights is not None:
        p = np.clip(np.asarray(prev_weights, dtype=float).reshape(-1), 0, None)
        turnover = float(np.abs(w - p / p.sum()).sum())

    rec = Recommendation(
        profile=prof.key, capital=float(capital),
        weights=pd.Series(w, index=list(ASSETS)),
        amounts=pd.Series(_to_cents(w, float(capital)), index=list(ASSETS)),
        portfolio_vol=vol, equity_cap_profile=prof.max_equity,
        equity_cap_effective=cap_eff, crisis_probability=float(crisis_probability),
        turnover=turnover, status=out.status, warnings=out.warnings,
    )
    rec.explanation = _explain(rec, prof)
    return rec


_ML1_READY = False


def recommend_for_date(date, capital: float, profile: str, prev_weights=None, *,
                       use_oos: bool = True, **kwargs) -> Recommendation:
    """Branche ML 2 (covariance annualisée) et ML 1 (p_crisis) pour une date donnée.
    ML 2 : predict_volatility(date) -> (vols: Series, cov: DataFrame)
    ML 1 : predict_crisis(date) -> float (nécessite le "release" ML 1 généré en local)."""
    from src.models.ml2.predict_volatility import predict_volatility
    from .ml1_loader import load_ml1

    predict_crisis = load_ml1()
    _vols, cov = predict_volatility(date, use_oos=use_oos)
    p = predict_crisis(date)
    return recommend(capital, profile, cov, p, prev_weights=prev_weights, **kwargs)