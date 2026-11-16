"""Entraînement et comparaison ML 2 (prévision de volatilité à 21 séances).

Usage : python src/models/ml2/train_ml2.py
Sorties : predictions.parquet, model_metrics.csv, best_model.pkl
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ASSETS, HORIZON, MODELS, OUT_DIR, Estimator, load_data,
                    score, target_col, walk_forward)

TEST_FRAC = 0.2   # derniers 20 % de la période hors-échantillon = test final


def main():
    X, Y = load_data()
    preds = {}
    for kind in MODELS:
        for a in ASSETS:
            print(f"walk-forward {kind:15s} {a}", flush=True)
            preds[(kind, a)] = walk_forward(X, Y, kind, a)

    oos = preds[(MODELS[0], ASSETS[0])].dropna().index
    cut = oos[int(len(oos) * (1 - TEST_FRAC))]
    # purge entre sélection et test : on saute 21 séances
    pos = X.index.get_loc(cut)
    sel_idx = oos[oos < X.index[pos - HORIZON]]
    test_idx = oos[oos >= cut]

    rows = []
    for kind in MODELS:
        for a in ASSETS:
            y = Y[target_col(a)]
            for periode, idx in (("selection", sel_idx), ("test", test_idx)):
                s = score(y.loc[idx].to_numpy(), preds[(kind, a)].loc[idx].to_numpy())
                rows.append({"modele": kind, "actif": a, "periode": periode, **s})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)

    # Meilleur modèle : QLIKE moyen (4 actifs) sur la période de SÉLECTION uniquement
    sel = metrics[metrics.periode == "selection"].groupby("modele")["QLIKE"].mean()
    best = sel.idxmin()
    print("\nQLIKE moyen (sélection):\n", sel.round(4), "\nMeilleur :", best)

    # Prédictions OOS du meilleur modèle (propres, sans fuite) pour le backtest
    out = pd.DataFrame({f"{a}": preds[(best, a)] for a in ASSETS}).dropna()
    out.columns = [f"pred_vol_{a}" for a in ASSETS]
    for kind in MODELS:  # on garde aussi les autres modèles pour comparaison
        for a in ASSETS:
            out[f"{kind}__{a}"] = preds[(kind, a)].loc[out.index]
    out.to_parquet(OUT_DIR / "predictions.parquet")

    # Modèle final : ré-entraîné sur tout l'historique disponible
    final = {a: Estimator(best, a).fit(X, Y[target_col(a)]) for a in ASSETS}
    joblib.dump({"kind": best, "models": final, "features": list(X.columns),
                 "oos_start": str(out.index[0].date()),
                 "test_start": str(test_idx[0].date())}, OUT_DIR / "best_model.pkl")
    print("Sauvegardé dans", OUT_DIR)


if __name__ == "__main__":
    main()