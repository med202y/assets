"""Utilitaires partagés ML 2 : chargement, modèles, métriques, walk-forward."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

RACINE = Path(__file__).resolve().parents[3]
PROCESSED = RACINE / "src" / "data" / "financial_dataset" / "processed"
OUT_DIR = Path(__file__).resolve().parent
REPORTS = RACINE / "reports" / "ml2"

ASSETS = ["SPY", "IEF", "LQD", "VNQ"]
HORIZON = 21          # séances (règle des 21 jours)
MIN_TRAIN = 756       # ~3 ans avant la première prédiction hors-échantillon
STEP = 126            # ré-entraînement tous les ~6 mois
EPS = 1e-4
MODELS = ["Historique_21j", "HAR", "RandomForest", "XGBoost"]


def load_data():
    X = pd.read_parquet(PROCESSED / "features_ml2.parquet").sort_index()
    Y = pd.read_parquet(PROCESSED / "targets_ml2.parquet").sort_index()
    assert X.index.equals(Y.index) and X.index.is_unique
    return X, Y


def target_col(a):
    return f"y_{a}_vol_21d_fwd"


def har_cols(a):
    """Composantes jour / semaine / mois du modèle HAR."""
    return [f"{a}_vol_1d", f"{a}_vol_5d", f"{a}_vol_21d"]


def _log(df):
    return np.log(df.clip(lower=0) + EPS)


class Estimator:
    """Un modèle (type, actif). Cible et variables de vol en log, retour en niveau."""

    def __init__(self, kind, asset):
        self.kind, self.asset = kind, asset
        self.model = None

    def _prep(self, X):
        if self.kind == "HAR":
            return _log(X[har_cols(self.asset)])
        vol_cols = [c for c in X.columns if "_vol_" in c or c in ("VIX", "mean_vol_21d")]
        Z = X.copy()
        Z[vol_cols] = _log(Z[vol_cols])
        return Z

    def fit(self, X, y):
        if self.kind == "Historique_21j":
            return self
        if self.kind == "HAR":
            self.model = LinearRegression()
        elif self.kind == "RandomForest":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=6, min_samples_leaf=20,
                max_features=0.5, n_jobs=-1, random_state=0)
        elif self.kind == "XGBoost":
            self.model = XGBRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0,
                n_jobs=-1, random_state=0)
        self.cols = list(self._prep(X).columns)
        self.model.fit(self._prep(X), np.log(y + EPS))
        return self

    def predict(self, X):
        if self.kind == "Historique_21j":
            return X[f"{self.asset}_vol_21d"].to_numpy()
        p = self.model.predict(self._prep(X)[self.cols])
        return np.exp(p) - EPS


# ---------- métriques ----------
def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def qlike(y, f):
    """QLIKE sur les variances (y², f²) : asymétrique, pénalise la sous-estimation."""
    r = (y ** 2 + EPS ** 2) / (f ** 2 + EPS ** 2)
    return float(np.mean(r - np.log(r) - 1))


def score(y, f):
    return {"RMSE": rmse(y, f), "MAE": mae(y, f), "QLIKE": qlike(y, f)}


# ---------- walk-forward ----------
def walk_forward(X, Y, kind, asset):
    """Prédictions hors-échantillon, fenêtre extensible, purge de 21 séances.

    Pour un bloc de test commençant à la ligne s, on n'entraîne que sur les
    lignes < s - HORIZON : leurs cibles (t+1..t+21) ne débordent pas sur le bloc.
    """
    y = Y[target_col(asset)]
    n = len(X)
    pred = pd.Series(np.nan, index=X.index)
    for s in range(MIN_TRAIN + HORIZON, n, STEP):
        e = min(s + STEP, n)
        tr = slice(0, s - HORIZON)
        est = Estimator(kind, asset).fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[s:e] = est.predict(X.iloc[s:e])
    return pred