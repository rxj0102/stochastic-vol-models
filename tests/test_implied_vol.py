"""Tests for implied-vol inversion, surfaces and smile analytics."""

import numpy as np

from implied_vol.arbitrage_checks import ArbitrageFreeChecker
from implied_vol.bsm_inversion import BSMInverter
from implied_vol.surface import IVSurface
from utils.statistics import bsm_price


def test_bsm_inversion_roundtrip():
    inv = BSMInverter()
    S0, K, T, r, q, sig = 100, 105, 0.75, 0.02, 0.01, 0.23
    price = bsm_price(S0, K, T, r, q, sig, "call")
    iv = inv.implied_vol(price, S0, K, T, r, q, "call")
    assert abs(iv - sig) < 1e-7
    assert abs(bsm_price(S0, K, T, r, q, iv, "call") - price) < 1e-8


def test_safe_iv_below_intrinsic_returns_nan():
    inv = BSMInverter()
    # A call priced below intrinsic by a wide margin is arbitrageable.
    iv = inv.safe_implied_vol(0.01, 100, 50, 1.0, 0.0, 0.0, "call")
    assert np.isnan(iv)


def test_calendar_spread_detection():
    chk = ArbitrageFreeChecker(spot=100.0)
    strikes = np.array([100.0])
    mats = np.array([0.5, 1.0])
    # Total variance decreasing in T: 0.3^2*0.5 > 0.18^2*1.0 -> violation.
    iv = np.array([[0.30], [0.18]])
    viol = chk.check_calendar_spread(strikes, mats, iv)
    assert viol[1, 0]


def test_butterfly_detection_on_concave_smile():
    chk = ArbitrageFreeChecker(spot=100.0, r=0.0, q=0.0)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    # Concave (inverted) smile: high in the middle -> negative density.
    iv_slice = np.array([0.15, 0.25, 0.35, 0.25, 0.15])
    viol, density = chk.check_butterfly(iv_slice, strikes, T=1.0)
    assert np.nansum(viol) > 0


def test_atm_vol_matches_model(base_params, market):
    S0, r, q = market["S0"], market["r"], market["q"]
    strikes = np.linspace(70, 130, 25)
    mats = np.array([0.25, 0.5, 1.0])
    surf = IVSurface.from_heston(base_params, S0, r, q, strikes, mats)
    # ATM-forward vol from interpolation should be finite and sensible.
    T = 1.0
    F = S0 * np.exp((r - q) * T)
    atm = surf.iv_at(F, T)
    assert 0.1 < atm < 0.4


# --- Additional coverage (audit remediation) -------------------------------


def test_safe_iv_above_maximum_returns_nan():
    inv = BSMInverter()
    # Call worth more than the discounted spot is arbitrageable -> NaN.
    iv = inv.safe_implied_vol(150.0, 100.0, 90.0, 1.0, 0.0, 0.0, "call")
    assert np.isnan(iv)


def test_batch_inversion_performance():
    import time

    inv = BSMInverter()
    rng = np.random.default_rng(0)
    n = 1000
    strikes = rng.uniform(80, 120, n)
    prices = np.array(
        [bsm_price(100, float(k), 1.0, 0.02, 0.0, 0.2, "call") for k in strikes]
    )
    t0 = time.perf_counter()
    ivs = inv.implied_vol_batch(prices, 100, strikes, 1.0, 0.02, 0.0, "call")
    elapsed = time.perf_counter() - t0
    assert np.nanmax(np.abs(ivs - 0.2)) < 1e-4
    assert elapsed < 2.0
