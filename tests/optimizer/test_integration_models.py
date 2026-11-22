"""Tests d'intégration avec les vrais modèles ML 1 / ML 2 ."""
import numpy as np
import pytest

from src.optimizer.recommender import recommend

DATE = "2024-06-28"


def test_real_ml2_covariance_feeds_optimizer():
    ml2 = pytest.importorskip("src.models.ml2.predict_volatility")
    try:
        vols, cov = ml2.predict_volatility(DATE)
    except FileNotFoundError:
        pytest.skip("artefacts ML 2 absents")
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > -1e-9
    rec = recommend(10_000, "balanced", cov, 0.27)
    assert rec.weights.sum() == pytest.approx(1.0)
    assert rec.portfolio_vol <= 0.10 + 1e-6


def test_real_ml1_release_gives_probability():
    from src.optimizer.ml1_loader import load_ml1
    try:
        predict_crisis = load_ml1()
    except FileNotFoundError:
        pytest.skip("release ML 1 absent (data/processed/ml1_inference_release)")
    assert 0.0 <= predict_crisis(DATE) <= 1.0