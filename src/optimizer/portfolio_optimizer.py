
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from .constraints import build_constraints
from .profiles import ASSETS, Profile

log = logging.getLogger(__name__)

DEFAULT_TURNOVER_PENALTY = 0.002   # coût par unité de rotation (L1), à calibrer
_SOLVERS = ("CLARABEL", "ECOS", "SCS")
_TOL = 1e-6


class OptimizationError(RuntimeError):
    pass


@dataclass
class OptimizationOutput:
    weights: np.ndarray
    status: str                      
    warnings: list[str] = field(default_factory=list)
    solver: str | None = None


def prepare_sigma(sigma) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Valide Sigma, force symétrie + semi-définie positive, renvoie (Sigma, L, warnings)."""
    warnings: list[str] = []
    if isinstance(sigma, pd.DataFrame):
        missing = [a for a in ASSETS if a not in sigma.index or a not in sigma.columns]
        if missing:
            raise ValueError(f"Sigma: actifs manquants {missing}")
        S = sigma.loc[list(ASSETS), list(ASSETS)].to_numpy(dtype=float)
    else:
        S = np.asarray(sigma, dtype=float)
    if S.shape != (len(ASSETS), len(ASSETS)):
        raise ValueError(f"Sigma doit être {len(ASSETS)}x{len(ASSETS)}, reçu {S.shape}")
    if not np.isfinite(S).all():
        raise ValueError("Sigma contient des NaN/inf")
    S = (S + S.T) / 2.0
    eigval, eigvec = np.linalg.eigh(S)
    if eigval.min() < -1e-8 * max(abs(eigval.max()), 1e-12):
        warnings.append("Sigma n'était pas semi-définie positive : projection effectuée.")
    eigval = np.clip(eigval, 0.0, None)
    S = (eigvec * eigval) @ eigvec.T
    L = eigvec * np.sqrt(eigval)
    return S, L, warnings


def historical_expected_returns(daily_returns: pd.DataFrame, shrink: float = 0.5) -> np.ndarray:
    """Rendement annualisé historique par actif, rétréci vers la moyenne des actifs
    (shrink=1 -> tous égaux ; 0 -> brut). À calculer UNIQUEMENT sur des données passées."""
    mu = daily_returns[list(ASSETS)].mean().to_numpy(dtype=float) * 252
    return (1.0 - shrink) * mu + shrink * mu.mean()


def _solve(problem: cp.Problem) -> str | None:
    installed = set(cp.installed_solvers())
    for name in _SOLVERS:
        if name not in installed:
            continue
        try:
            problem.solve(solver=name)
        except (cp.SolverError, Exception) as exc:  # noqa: BLE001
            log.warning("Solveur %s en échec: %s", name, exc)
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return name
        if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
            return None
    return None


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.where(w < _TOL, 0.0, w)
    return w / w.sum()


def _defensive(profile: Profile) -> np.ndarray:
    ief = min(max(profile.min_gov_bonds, 0.6), 1.0)
    w = np.zeros(len(ASSETS))
    w[ASSETS.index("IEF")] = ief
    w[ASSETS.index("LQD")] = 1.0 - ief
    return w


def optimize_portfolio(
    sigma,
    profile: Profile,
    equity_cap: float,
    *,
    prev_weights=None,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    expected_returns=None,
    risk_aversion: float = 1.0,
    max_weight: float | None = None,
    max_turnover: float | None = None,
) -> OptimizationOutput:
    """min  gamma * w'Σw [- mu'w] + lambda * ||w - w_prev||_1
    Si expected_returns est None -> variance minimale (gamma = 1)."""
    S, L, warnings = prepare_sigma(sigma)
    n = len(ASSETS)

    prev = None
    if prev_weights is not None:
        prev = np.asarray(prev_weights, dtype=float).reshape(-1)
        if prev.shape != (n,) or not np.isfinite(prev).all() or prev.sum() <= 0:
            raise ValueError("prev_weights invalide")
        prev = np.clip(prev, 0.0, None)
        prev = prev / prev.sum()

    mu = None
    if expected_returns is not None:
        if isinstance(expected_returns, pd.Series):
            expected_returns = expected_returns.reindex(list(ASSETS)).to_numpy(dtype=float)
        mu = np.asarray(expected_returns, dtype=float).reshape(-1)
        if mu.shape != (n,) or not np.isfinite(mu).all():
            raise ValueError("expected_returns invalide")

    # Relaxations successives si le problème est infaisable (du moins au plus grave)
    attempts = [
        ("optimal", dict(enforce_vol=True, max_weight=max_weight, max_turnover=max_turnover)),
        ("relaxed_turnover", dict(enforce_vol=True, max_weight=max_weight, max_turnover=None)),
        ("relaxed_max_weight", dict(enforce_vol=True, max_weight=None, max_turnover=None)),
        ("relaxed_vol", dict(enforce_vol=False, max_weight=None, max_turnover=None)),
    ]
    seen = set()
    for status, opts in attempts:
        key = tuple(sorted((k, str(v)) for k, v in opts.items()))
        if key in seen:
            continue
        seen.add(key)

        use_mu = mu if status != "relaxed_vol" else None   # vol relâchée => variance minimale pure
        w = cp.Variable(n)
        variance = cp.sum_squares(L.T @ w)
        objective = (risk_aversion if use_mu is not None else 1.0) * variance
        if use_mu is not None:
            objective = objective - use_mu @ w
        if prev is not None and turnover_penalty > 0:
            objective = objective + turnover_penalty * cp.norm1(w - prev)
        cons = build_constraints(w, L, profile, equity_cap, prev_weights=prev, **opts)
        solver = _solve(cp.Problem(cp.Minimize(objective), cons))
        if solver is not None and w.value is not None:
            if status != "optimal":
                warnings.append(f"Contrainte(s) relâchée(s) car infaisable : {status}.")
            return OptimizationOutput(_clean(np.asarray(w.value).ravel()), status, warnings, solver)

    warnings.append("Aucune solution trouvée : allocation défensive de secours.")
    return OptimizationOutput(_defensive(profile), "fallback_defensive", warnings, None)