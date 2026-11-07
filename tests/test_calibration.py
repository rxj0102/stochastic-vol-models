"""Tests for the calibration pipeline (kept small for speed)."""

import numpy as np
import pytest

from calibration.heston_calibration import HestonCalibrator
from calibration.loss_functions import CalibrationLoss
from calibration.market_data import MarketIVSurface
from models.heston import HestonParams


@pytest.fixture(scope="module")
def synthetic():
    true = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.6, V0=0.045)
    S0, r, q = 100.0, 0.02, 0.0
    strikes = np.array([85, 95, 100, 105, 115.0])
    mats = np.array([0.25, 1.0])
    surf = MarketIVSurface.generate_heston_surface(true, S0, r, q, strikes, mats)
    return true, surf, S0, r, q


def test_loss_functions_basic():
    loss = CalibrationLoss()
    market = np.array([0.20, 0.22, 0.25])
    model = np.array([0.21, 0.22, 0.24])
    assert loss.rmse(market, model) > 0
    assert loss.rmse(market, market) == 0.0
    # NaNs are ignored.
    assert np.isfinite(loss.rmse([0.2, np.nan], [0.21, 0.3]))


@pytest.mark.slow
def test_synthetic_roundtrip(synthetic):
    true, surf, S0, r, q = synthetic
    cal = HestonCalibrator(seed=0)
    res = cal.calibrate_joint(surf, S0, r, q, loss_type="rmse", de_maxiter=40)
    assert res.rmse_bps < 50  # < 0.5 vol points
    # rho sign recovered.
    assert res.params.rho < 0
    for k in true.as_dict():
        rel = abs(res.params.as_dict()[k] - true.as_dict()[k]) / abs(true.as_dict()[k])
        assert rel < 0.30, f"{k} rel err {rel:.2%}"


def test_rmse_decreases_with_local_refinement(synthetic):
    """A short global search followed by local refine should fit well."""
    true, surf, S0, r, q = synthetic
    cal = HestonCalibrator(seed=1)
    res = cal.calibrate_joint(surf, S0, r, q, loss_type="rmse", de_maxiter=25)
    assert res.rmse_bps < 100
