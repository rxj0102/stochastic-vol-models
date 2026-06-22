"""Tests for the SABR model parameter validation and Hagan ATM limit."""

import numpy as np
import pytest

from models.sabr import SABRParams, hagan_lognormal_vol


def test_sabr_beta_out_of_range_raises():
    with pytest.raises(ValueError):
        SABRParams(alpha=0.2, beta=1.5, rho=-0.3, nu=0.5)
    with pytest.raises(ValueError):
        SABRParams(alpha=0.2, beta=-0.1, rho=-0.3, nu=0.5)


def test_sabr_invalid_alpha_nu_rho_raise():
    with pytest.raises(ValueError):
        SABRParams(alpha=-0.1, beta=0.5, rho=-0.3, nu=0.5)
    with pytest.raises(ValueError):
        SABRParams(alpha=0.2, beta=0.5, rho=1.2, nu=0.5)


def test_hagan_atm_limit_is_finite_and_smooth():
    F, T = 100.0, 1.0
    # Straddle the forward; the z/x(z) ratio must -> 1 at K = F with no blow-up.
    K = np.array([99.0, 99.9, 100.0, 100.1, 101.0])
    vol = hagan_lognormal_vol(F, K, T, alpha=0.2, beta=0.5, rho=-0.3, nu=0.5)
    assert np.all(np.isfinite(vol))
    # Continuity across the ATM point (no spike at K = F).
    assert abs(vol[2] - 0.5 * (vol[1] + vol[3])) < 5e-3
