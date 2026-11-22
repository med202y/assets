from __future__ import annotations

import cvxpy as cp
import numpy as np

from .profiles import ASSETS, EQUITY_ASSETS, GOV_BOND_ASSETS, Profile

EQUITY_IDX = [ASSETS.index(a) for a in EQUITY_ASSETS]
GOV_IDX = [ASSETS.index(a) for a in GOV_BOND_ASSETS]


def effective_equity_cap(profile_cap: float, crisis_probability: float, alpha: float = 1.0) -> float:
    """Plafond actions effectif = plafond * (1 - alpha * P_crise), borné à [0, plafond]."""
    if not 0.0 <= crisis_probability <= 1.0:
        raise ValueError(f"crisis_probability doit être dans [0, 1], reçu {crisis_probability}")
    if alpha < 0:
        raise ValueError("alpha doit être >= 0")
    factor = min(max(1.0 - alpha * crisis_probability, 0.0), 1.0)
    return profile_cap * factor


def build_constraints(
    w: cp.Variable,
    risk_factor: np.ndarray,
    profile: Profile,
    equity_cap: float,
    *,
    enforce_vol: bool = True,
    max_weight: float | None = None,
    prev_weights: np.ndarray | None = None,
    max_turnover: float | None = None,
) -> list:
    """risk_factor L tel que Sigma = L L'  =>  variance = ||L' w||^2 (convexe, robuste)."""
    cons = [cp.sum(w) == 1, w >= 0]
    cons.append(cp.sum(w[EQUITY_IDX]) <= equity_cap)
    cons.append(cp.sum(w[GOV_IDX]) >= profile.min_gov_bonds)
    if enforce_vol:
        cons.append(cp.sum_squares(risk_factor.T @ w) <= profile.max_vol ** 2)
    if max_weight is not None:
        cons.append(w <= max_weight)
    if max_turnover is not None and prev_weights is not None:
        cons.append(cp.norm1(w - prev_weights) <= max_turnover)
    return cons