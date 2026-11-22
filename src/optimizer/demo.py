"""Démo / vérification manuelle de l'optimiseur.

    python -m src.optimizer.demo                        # covariance illustrative
    python -m src.optimizer.demo --date 2024-06-28      # vraie covariance ML 2 (+ ML 1 si dispo)
    python -m src.optimizer.demo --profile aggressive --capital 25000 --crisis 0.4
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .profiles import ASSETS, PROFILES, get_profile
from .recommender import recommend


def synthetic_cov() -> pd.DataFrame:
    vols = np.array([0.18, 0.07, 0.09, 0.16])
    corr = np.array([[1.0, -0.20, 0.30, 0.75], [-0.20, 1.0, 0.70, -0.05],
                     [0.30, 0.70, 1.0, 0.35], [0.75, -0.05, 0.35, 1.0]])
    return pd.DataFrame(np.diag(vols) @ corr @ np.diag(vols), index=ASSETS, columns=ASSETS)


def check(rec) -> None:
    """Vérifie les contraintes du profil sur la recommandation (lève AssertionError sinon)."""
    prof = get_profile(rec.profile)
    w = rec.weights
    assert abs(w.sum() - 1) < 1e-6, "somme des poids != 1"
    assert (w >= -1e-9).all(), "poids négatif"
    assert w[["SPY", "VNQ"]].sum() <= rec.equity_cap_effective + 1e-6, "plafond actions dépassé"
    assert w["IEF"] >= prof.min_gov_bonds - 1e-6, "plancher obligations d'État non respecté"
    if rec.status == "optimal":
        assert rec.portfolio_vol <= prof.max_vol + 1e-6, "volatilité max dépassée"
    assert abs(rec.amounts.sum() - rec.capital) < 0.011, "montants != capital"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--capital", type=float, default=10_000)
    ap.add_argument("--profile", default="balanced", choices=sorted(PROFILES))
    ap.add_argument("--crisis", type=float, default=0.27,
                    help="p_crisis manuelle (démo uniquement ; ignorée si ML 1 disponible)")
    ap.add_argument("--date", default=None, help="AAAA-MM-JJ : utilise les vrais modèles")
    args = ap.parse_args()

    crisis, source = args.crisis, "manuelle (démo)"
    if args.date is None:
        cov = synthetic_cov()
        print("Covariance : illustrative (pas de ML 2)")
    else:
        from src.models.ml2.predict_volatility import predict_volatility
        vols, cov = predict_volatility(args.date)
        print(f"Covariance : ML 2 au {args.date}\nVolatilités prédites :\n{vols.round(3).to_string()}")
        try:
            from .ml1_loader import load_ml1
            crisis, source = load_ml1()(args.date), "ML 1"
        except Exception as exc:  # noqa: BLE001
            print(f"[!] ML 1 indisponible ({type(exc).__name__}: {exc}) -> p_crisis manuelle = {crisis}")

    rec = recommend(capital=args.capital, profile=args.profile, sigma=cov,
                    crisis_probability=crisis, prev_weights=None, alpha=1.0)
    print(f"\np_crisis = {rec.crisis_probability:.1%} (source : {source})")
    print("\nPoids :\n" + rec.weights.round(4).to_string())
    print("\nMontants :\n" + rec.amounts.to_string())
    print(f"\nVolatilité du portefeuille : {rec.portfolio_vol:.2%}   statut : {rec.status}")
    print("\n" + rec.explanation)
    check(rec)
    print("\n✔ Toutes les contraintes du profil sont respectées.")


if __name__ == "__main__":
    main()