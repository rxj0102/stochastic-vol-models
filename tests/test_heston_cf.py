"""Tests for the Heston characteristic function."""

import numpy as np

from characteristic_functions.heston_cf import HestonCharacteristicFunction


def _cf(params):
    return HestonCharacteristicFunction("little_trap")


def test_cf_at_zero_is_one(base_params, market):
    cf = HestonCharacteristicFunction()
    p = base_params
    val = cf.cf(
        np.array([0.0]),
        market["S0"],
        p.V0,
        market["T"],
        market["r"],
        market["q"],
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
    )[0]
    assert abs(val - 1.0) < 1e-12


def test_cf_modulus_bounded(base_params, market):
    cf = HestonCharacteristicFunction()
    p = base_params
    u = np.linspace(-300, 300, 6001)
    vals = cf.cf(
        u,
        market["S0"],
        p.V0,
        market["T"],
        market["r"],
        market["q"],
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
    )
    assert np.max(np.abs(vals)) <= 1.0 + 1e-9


def test_martingale_condition(base_params, market):
    """phi(-i) must equal the forward price (risk-neutral martingale)."""
    cf = HestonCharacteristicFunction()
    p = base_params
    val = cf.cf(
        np.array([-1j]),
        market["S0"],
        p.V0,
        market["T"],
        market["r"],
        market["q"],
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
    )[0]
    fwd = market["S0"] * np.exp((market["r"] - market["q"]) * market["T"])
    assert abs(val / fwd - 1.0) < 1e-10


def test_formulations_agree(base_params, market):
    cf = HestonCharacteristicFunction()
    p = base_params
    u = np.linspace(1.0, 80.0, 200)
    v1 = cf.cf(
        u,
        market["S0"],
        p.V0,
        market["T"],
        market["r"],
        market["q"],
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
        formulation="little_trap",
    )
    v2 = cf.cf(
        u,
        market["S0"],
        p.V0,
        market["T"],
        market["r"],
        market["q"],
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
        formulation="original",
    )
    assert np.max(np.abs(v1 - v2)) < 1e-10


def test_log_return_mean(base_params, market):
    cf = HestonCharacteristicFunction()
    p = base_params
    m = cf.log_return_moments(
        market["T"], market["r"], market["q"], p.kappa, p.theta, p.xi, p.rho, p.V0
    )
    # E[ln(S_T/S0)] = (r - q) T - 0.5 E[int V dt]; here ~ 0.
    ekt = np.exp(-p.kappa * market["T"])
    eiv = p.theta * market["T"] + (p.V0 - p.theta) * (1 - ekt) / p.kappa
    expected = (market["r"] - market["q"]) * market["T"] - 0.5 * eiv
    assert abs(m["mean"] - expected) < 1e-3


def test_skewness_negative_for_negative_rho(base_params, market):
    cf = HestonCharacteristicFunction()
    p = base_params  # rho = -0.7
    m = cf.log_return_moments(
        market["T"], market["r"], market["q"], p.kappa, p.theta, p.xi, p.rho, p.V0
    )
    assert m["skewness"] < 0.0


def test_excess_kurtosis_positive(base_params, market):
    cf = HestonCharacteristicFunction()
    p = base_params
    m = cf.log_return_moments(
        market["T"], market["r"], market["q"], p.kappa, p.theta, p.xi, p.rho, p.V0
    )
    assert m["kurtosis"] > 0.0
