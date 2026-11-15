"""Figures de comparaison ML 2. Usage : python src/models/ml2/evaluate_ml2.py"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS, MODELS, OUT_DIR, REPORTS, load_data, target_col


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(OUT_DIR / "model_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, per in enumerate(["selection", "test"]):
        for j, met in enumerate(["RMSE", "MAE", "QLIKE"]):
            t = m[m.periode == per].pivot(index="actif", columns="modele", values=met)[MODELS]
            t.plot.bar(ax=axes[i, j], rot=0, legend=(i == 0 and j == 0))
            axes[i, j].set_title(f"{met} – {per}")
    plt.tight_layout(); plt.savefig(REPORTS / "model_comparison.png", dpi=130); plt.close()

    _, Y = load_data()
    p = pd.read_parquet(OUT_DIR / "predictions.parquet")
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, a in zip(axes, ASSETS):
        ax.plot(Y.loc[p.index, target_col(a)], color="black", lw=0.8, label="Réalisée (21 j suivants)")
        ax.plot(p[f"pred_vol_{a}"], color="tab:blue", lw=0.8, label="Prédite")
        ax.set_title(a); ax.set_ylabel("vol. annualisée")
    axes[0].legend(); plt.tight_layout()
    plt.savefig(REPORTS / "volatility_predictions.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()