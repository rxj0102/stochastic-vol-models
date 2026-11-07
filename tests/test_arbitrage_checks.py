"""Tests for static arbitrage detection on IV surfaces."""

import numpy as np

from implied_vol.arbitrage_checks import ArbitrageFreeChecker
from implied_vol.surface import IVSurface


def test_heston_surface_arbitrage_free(base_params, market):
    S0, r, q = market["S0"], market["r"], market["q"]
    strikes = np.linspace(80, 120, 21)
    mats = np.array([0.25, 0.5, 1.0, 2.0])
    surf = IVSurface.from_heston(base_params, S0, r, q, strikes, mats)
    chk = ArbitrageFreeChecker(spot=S0, r=r, q=q)
    report = chk.full_arbitrage_report(strikes, mats, surf.iv)
    assert report["calendar_violations"] == 0
    assert report["butterfly_violations"] == 0


def test_perturbed_surface_calendar_violation(base_params, market):
    S0, r, q = market["S0"], market["r"], market["q"]
    strikes = np.linspace(80, 120, 21)
    mats = np.array([0.25, 0.5, 1.0])
    surf = IVSurface.from_heston(base_params, S0, r, q, strikes, mats)
    iv = surf.iv.copy()
    # Collapse the long-maturity vols so total variance falls in T.
    iv[-1, :] *= 0.4
    chk = ArbitrageFreeChecker(spot=S0, r=r, q=q)
    viol = chk.check_calendar_spread(strikes, mats, iv)
    assert viol.sum() > 0


def test_butterfly_violation_handcrafted():
    chk = ArbitrageFreeChecker(spot=100.0)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    iv_slice = np.array([0.15, 0.30, 0.45, 0.30, 0.15])  # strongly concave
    viol, density = chk.check_butterfly(iv_slice, strikes, T=1.0)
    assert np.nansum(viol) > 0


def test_risk_neutral_density_nonnegative(base_params, market):
    S0, r, q = market["S0"], market["r"], market["q"]
    strikes = np.linspace(70, 130, 41)
    surf = IVSurface.from_heston(base_params, S0, r, q, strikes, np.array([1.0]))
    chk = ArbitrageFreeChecker(spot=S0, r=r, q=q)
    _, density = chk.check_butterfly(surf.iv[0], strikes, T=1.0)
    finite = density[np.isfinite(density)]
    assert np.all(finite > -1e-6)
