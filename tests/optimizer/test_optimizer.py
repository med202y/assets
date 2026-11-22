import numpy as np
import pandas as pd
import pytest

from src.optimizer.constraints import effective_equity_cap
from src.optimizer.portfolio_optimizer import historical_expected_returns, prepare_sigma
from src.optimizer.profiles import ASSETS, PROFILES, get_profile
from src.optimizer.recommender import recommend

VOLS = np.array([0.18, 0.07, 0.09, 0.16])
CORR = np.array([
    [1.0, -0.20, 0.30, 0.75],
    [-0.20, 1.0, 0.70, -0.05],
    [0.30, 0.70, 1.0, 0.35],
    [0.75, -0.05, 0.35, 1.0],
])
SIGMA = np.diag(VOLS) @ CORR @ np.diag(VOLS)
MU_EQUITY = np.array([0.15, 0.02, 0.03, 0.10])   # force les actions à saturer le plafond


def eq(rec):
    return rec.weights[["SPY", "VNQ"]].sum()


def test_profiles_match_spec():
    c = PROFILES["conservative"]
    assert (c.max_vol, c.max_equity, c.min_gov_bonds) == (0.06, 0.25, 0.30)
    assert get_profile("Équilibré").key == "balanced"
    with pytest.raises(ValueError):
        get_profile("inconnu")


def test_effective_cap_formula():
    assert effective_equity_cap(0.50, 0.27, 1.0) == pytest.approx(0.365)
    assert effective_equity_cap(0.50, 1.0, 1.0) == 0.0
    assert effective_equity_cap(0.50, 0.5, 0.0) == 0.50
    with pytest.raises(ValueError):
        effective_equity_cap(0.5, 1.2)


@pytest.mark.parametrize("profile", list(PROFILES))
@pytest.mark.parametrize("p", [0.0, 0.27, 0.8])
def test_constraints_respected(profile, p):
    rec = recommend(10_000, profile, SIGMA, p)
    prof = PROFILES[profile]
    assert rec.weights.sum() == pytest.approx(1.0)
    assert (rec.weights >= -1e-9).all()
    assert eq(rec) <= rec.equity_cap_effective + 1e-6
    assert rec.weights["IEF"] >= prof.min_gov_bonds - 1e-6
    if rec.status == "optimal":
        assert rec.portfolio_vol <= prof.max_vol + 1e-6
    else:   # ex. prudent + forte crise : 6 % de vol inatteignable avec <= 5 % d'actions
        assert rec.status == "relaxed_vol" and rec.warnings


def test_amounts_sum_to_capital_exactly():
    rec = recommend(10_000.01, "balanced", SIGMA, 0.27)
    assert round(rec.amounts.sum(), 2) == 10_000.01


def test_cap_binds_and_crisis_reduces_equity():
    r0 = recommend(10_000, "balanced", SIGMA, 0.0, expected_returns=MU_EQUITY, risk_aversion=2.0)
    r1 = recommend(10_000, "balanced", SIGMA, 0.27, expected_returns=MU_EQUITY, risk_aversion=2.0)
    assert eq(r0) == pytest.approx(0.50, abs=1e-4)
    assert eq(r1) == pytest.approx(0.365, abs=1e-4)
    assert eq(r1) < eq(r0)


def test_turnover_penalty_keeps_previous_weights():
    prev = np.array([0.25, 0.40, 0.25, 0.10])
    free = recommend(10_000, "balanced", SIGMA, 0.0, prev_weights=prev, turnover_penalty=0.0)
    sticky = recommend(10_000, "balanced", SIGMA, 0.0, prev_weights=prev, turnover_penalty=1.0)
    assert sticky.turnover < free.turnover
    assert sticky.turnover < 1e-3


def test_max_turnover_hard_limit():
    prev = recommend(10_000, "conservative", SIGMA, 0.0).weights.to_numpy()
    rec = recommend(10_000, "conservative", SIGMA, 0.0, prev_weights=prev,
                    expected_returns=MU_EQUITY, risk_aversion=2.0, max_turnover=0.05)
    assert rec.status == "optimal"
    assert rec.turnover <= 0.05 + 1e-6


def test_infeasible_vol_is_relaxed_not_crashing():
    hot = SIGMA * 25   # vols x5 : même la variance minimale dépasse 6 %
    rec = recommend(10_000, "conservative", hot, 0.1)
    assert rec.status == "relaxed_vol"
    assert rec.warnings
    assert rec.weights.sum() == pytest.approx(1.0)


def test_non_psd_sigma_is_projected():
    bad = SIGMA.copy()
    bad[0, 1] = bad[1, 0] = 0.5   # corrélation impossible
    _, _, warns = prepare_sigma(bad)
    assert warns
    assert recommend(10_000, "balanced", bad, 0.1).weights.sum() == pytest.approx(1.0)


def test_dataframe_sigma_is_reordered():
    df = pd.DataFrame(SIGMA, index=ASSETS, columns=ASSETS)
    order = ["VNQ", "SPY", "LQD", "IEF"]
    a = recommend(10_000, "balanced", df, 0.2)
    b = recommend(10_000, "balanced", df.loc[order, order], 0.2)
    assert np.allclose(a.weights, b.weights, atol=1e-6)


@pytest.mark.parametrize("kwargs", [
    dict(capital=0), dict(capital=-5), dict(profile="x"), dict(crisis_probability=1.5),
])
def test_invalid_inputs(kwargs):
    args = dict(capital=1000, profile="balanced", sigma=SIGMA, crisis_probability=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        recommend(**args)


def test_profiles_differ_with_expected_returns():
    eqs = [eq(recommend(10_000, p, SIGMA, 0.0, expected_returns=MU_EQUITY, risk_aversion=2.0))
           for p in ["conservative", "balanced", "aggressive"]]
    assert eqs[0] < eqs[1] < eqs[2]


def test_historical_expected_returns_shape():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(0.0004, 0.01, (500, 4)), columns=ASSETS)
    assert historical_expected_returns(df).shape == (4,)