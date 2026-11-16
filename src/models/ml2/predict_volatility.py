"""Fonction de prédiction ML 2 : volatilités à 21 séances + matrice de covariance.

    from predict_volatility import predict_volatility
    vols, cov = predict_volatility("2024-06-28")
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ASSETS, OUT_DIR, PROCESSED, load_data  # noqa: E402

CORR_WINDOW = 252


def _returns():
    px = pd.read_csv(PROCESSED / "final_dataset.csv", index_col=0, parse_dates=True)
    return px[ASSETS].sort_index().pct_change().dropna()


def make_psd(cov, tol=1e-10):
    """Projette sur le cône semi-défini positif (valeurs propres >= tol)."""
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    return (v * np.clip(w, tol, None)) @ v.T


def is_psd(cov, tol=1e-10):
    return bool(np.linalg.eigvalsh((cov + cov.T) / 2).min() >= -tol)


def covariance_matrix(vols, date, returns=None):
    """Σ = D · C · D ; C = corrélation des CORR_WINDOW séances précédant `date`."""
    r = _returns() if returns is None else returns
    hist = r.loc[:date].iloc[-CORR_WINDOW:]
    C = hist.corr().loc[ASSETS, ASSETS].to_numpy()
    D = np.diag(vols.loc[ASSETS].to_numpy())
    cov = D @ C @ D                      # annualisée, comme les volatilités
    if not is_psd(cov):
        cov = make_psd(cov)
    return pd.DataFrame(cov, index=ASSETS, columns=ASSETS)


def predict_volatility(date, use_oos=True):
    """Retourne (vols, cov) pour `date` (info disponible à la clôture de `date`).

    use_oos=True : prédiction hors-échantillon du walk-forward quand elle existe
    (à utiliser dans le backtest) ; sinon, le modèle final ré-entraîné.
    """
    date = pd.Timestamp(date)
    X, _ = load_data()
    if date not in X.index:
        raise KeyError(f"{date.date()} absent de features_ml2 "
                       f"({X.index[0].date()} à {X.index[-1].date()})")
    pred_path = OUT_DIR / "predictions.parquet"
    if use_oos and pred_path.exists():
        p = pd.read_parquet(pred_path)
        if date in p.index:
            vols = pd.Series({a: p.loc[date, f"pred_vol_{a}"] for a in ASSETS})
            return vols, covariance_matrix(vols, date)
    bundle = joblib.load(OUT_DIR / "best_model.pkl")
    row = X.loc[[date]]
    vols = pd.Series({a: float(bundle["models"][a].predict(row)[0]) for a in ASSETS})
    return vols, covariance_matrix(vols, date)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "2026-10-05"
    v, c = predict_volatility(d, use_oos=False)
    print(v.round(4)); print(c.round(5)); print("PSD:", is_psd(c.to_numpy()))