"""Tests for the Fourier pricing methods."""

import numpy as np

from fourier_pricing.carr_madan import CarrMadanPricer
from fourier_pricing.cos_method import COSMethodPricer
from fourier_pricing.lewis_method import LewisMethodPricer
from implied_vol.bsm_inversion import BSMInverter
from models.heston import HestonParams
from utils.statistics import bsm_price


def test_carr_madan_lower_bound(base_params, market):
    cm = CarrMadanPricer()
    S0, K, T, r, q = market["S0"], 100.0, market["T"], market["r"], market["q"]
    price = cm.price(S0, K, T, r, q, base_params, "call")
    lb = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    assert price >= lb - 1e-8


def test_cos_matches_carr_madan(base_params, market):
    cos, cm = COSMethodPricer(), CarrMadanPricer()
    inv = BSMInverter()
    S0, T, r, q = market["S0"], market["T"], market["r"], market["q"]
    for K in [80, 90, 100, 110, 120]:
        iv_cos = inv.implied_vol(cos.price(S0, K, T, r, q, base_params), S0, K, T, r, q)
        iv_cm = inv.implied_vol(cm.price(S0, K, T, r, q, base_params), S0, K, T, r, q)
        assert abs(iv_cos - iv_cm) < 1e-3


def test_lewis_matches_carr_madan(base_params, market):
    lew, cm = LewisMethodPricer(), CarrMadanPricer()
    inv = BSMInverter()
    S0, T, r, q = market["S0"], market["T"], market["r"], market["q"]
    for K in [85, 100, 115]:
        iv_lew = inv.implied_vol(lew.price(S0, K, T, r, q, base_params), S0, K, T, r, q)
        iv_cm = inv.implied_vol(cm.price(S0, K, T, r, q, base_params), S0, K, T, r, q)
        assert abs(iv_lew - iv_cm) < 1e-3


def test_put_call_parity_carr_madan(base_params, market):
    cm = CarrMadanPricer()
    S0, K, T, r, q = market["S0"], 100.0, market["T"], market["r"], market["q"]
    c = cm.price(S0, K, T, r, q, base_params, "call")
    p = cm.price(S0, K, T, r, q, base_params, "put")
    parity = c - p - (S0 * np.exp(-q * T) - K * np.exp(-r * T))
    assert abs(parity) < 1e-6


def test_put_call_parity_cos(base_params, market):
    cos = COSMethodPricer()
    S0, K, T, r, q = market["S0"], 100.0, market["T"], market["r"], market["q"]
    c = cos.price(S0, K, T, r, q, base_params, "call", N=256)
    p = cos.price(S0, K, T, r, q, base_params, "put", N=256)
    parity = c - p - (S0 * np.exp(-q * T) - K * np.exp(-r * T))
    assert abs(parity) < 1e-3


def test_heston_reduces_to_bsm(market):
    """As xi -> 0, rho -> 0 the Heston price collapses to Black-Scholes."""
    cos = COSMethodPricer()
    inv = BSMInverter()
    S0, T, r, q = market["S0"], market["T"], market["r"], market["q"]
    sig = 0.20
    p = HestonParams(kappa=2.0, theta=sig**2, xi=0.02, rho=0.0, V0=sig**2)
    price = cos.price(S0, 100.0, T, r, q, p, "call")
    iv = inv.implied_vol(price, S0, 100.0, T, r, q)
    assert abs(iv - sig) < 1e-2


def test_cos_exponential_convergence(base_params, market):
    cos = COSMethodPricer()
    res = cos.convergence_test(
        market["S0"], 100.0, market["T"], market["r"], market["q"], base_params
    )
    err = res["abs_error"]
    # Error should drop by orders of magnitude from N=16 to N=128.
    assert err[1] > err[4] * 50


def test_fft_alpha_invariance(base_params, market):
    cm = CarrMadanPricer()
    S0, K, T, r, q = market["S0"], 100.0, market["T"], market["r"], market["q"]
    prices = [
        cm.price(S0, K, T, r, q, base_params, "call", alpha=a)
        for a in (1.1, 1.5, 2.0, 2.5)
    ]
    assert max(prices) - min(prices) < 1e-3


# --- Robustness guards (added in audit remediation) ------------------------


def test_carr_madan_rejects_nonpositive_alpha(base_params, market):
    import pytest

    cm = CarrMadanPricer()
    S0, T, r, q = market["S0"], market["T"], market["r"], market["q"]
    for bad_alpha in (0.0, -0.5):
        with pytest.raises(ValueError):
            cm.price(S0, 100.0, T, r, q, base_params, "call", alpha=bad_alpha)


def test_carr_madan_rejects_non_power_of_two_N(base_params, market):
    import pytest

    cm = CarrMadanPricer()
    with pytest.raises(ValueError):
        cm.price_fft(
            market["S0"], market["T"], market["r"], market["q"], base_params, N=1000
        )


def test_cf_rejects_negative_variance_params(market):
    import pytest

    from characteristic_functions.heston_cf import HestonCharacteristicFunction

    cf = HestonCharacteristicFunction()
    with pytest.raises(ValueError):
        cf.cf(
            np.array([1.0]),
            market["S0"],
            -0.04,
            market["T"],
            market["r"],
            market["q"],
            2.0,
            0.04,
            0.5,
            -0.7,
        )
