"""Tests for the Heston simulation schemes."""

import numpy as np

from diagnostics.correlation_analysis import CorrelationEffects
from simulation.heston_euler import HestonEulerSimulator
from simulation.heston_qe import HestonQESimulator


def test_qe_martingale(base_params, market):
    qe = HestonQESimulator()
    S0, T, r, q = market["S0"], market["T"], market["r"], market["q"]
    S, _ = qe.simulate(S0, base_params.V0, T, r, q, base_params, 200_000, 50, seed=1)
    ST = S[:, -1]
    fwd = S0 * np.exp((r - q) * T)
    stderr = ST.std() / np.sqrt(len(ST))
    assert abs(ST.mean() - fwd) < 3 * stderr


def test_qe_variance_moment(base_params, market):
    qe = HestonQESimulator()
    p, T = base_params, market["T"]
    _, V = qe.simulate(100, p.V0, T, market["r"], market["q"], p, 200_000, 50, seed=2)
    EVT = p.theta + (p.V0 - p.theta) * np.exp(-p.kappa * T)
    stderr = V[:, -1].std() / np.sqrt(V.shape[0])
    assert abs(V[:, -1].mean() - EVT) < 4 * stderr


def test_qe_variance_nonnegative(base_params, market):
    qe = HestonQESimulator()
    # Deliberately violate Feller to stress the scheme.
    from models.heston import HestonParams

    p = HestonParams(kappa=1.0, theta=0.04, xi=0.9, rho=-0.7, V0=0.04)
    _, V = qe.simulate(100, p.V0, 1.0, 0.0, 0.0, p, 50_000, 100, seed=3)
    assert V.min() >= 0.0


def test_euler_can_go_negative_without_truncation(base_params):
    """Raw Euler variance updates can dip below zero (motivating truncation)."""
    rng = np.random.default_rng(0)
    p = base_params
    dt = 1.0 / 252
    v = 0.001  # near the boundary
    went_negative = False
    for _ in range(100000):
        dWv = np.sqrt(dt) * rng.standard_normal()
        v_new = v + p.kappa * (p.theta - v) * dt + p.xi * np.sqrt(max(v, 0)) * dWv
        if v_new < 0:
            went_negative = True
            break
    assert went_negative


def test_antithetic_variance_reduction(base_params, market):
    eul = HestonEulerSimulator()
    S0, T, r, q = market["S0"], market["T"], market["r"], market["q"]
    K = 100.0

    def mc_price(antithetic):
        S, _ = eul.simulate(
            S0,
            base_params.V0,
            T,
            r,
            q,
            base_params,
            20_000,
            50,
            seed=7,
            antithetic=antithetic,
        )
        payoff = np.exp(-r * T) * np.maximum(S[:, -1] - K, 0.0)
        return payoff

    plain = mc_price(False)
    anti = mc_price(True)
    # Compare standard error of the mean.
    se_plain = plain.std() / np.sqrt(len(plain))
    # Antithetic: pair means.
    n = len(anti) // 2
    paired = 0.5 * (anti[:n] + anti[n : 2 * n])
    se_anti = paired.std() / np.sqrt(len(paired))
    assert se_plain / se_anti > 1.2


def test_leverage_effect_recovers_rho(base_params, market):
    qe = HestonQESimulator()
    S, V = qe.simulate(
        100,
        base_params.V0,
        1.0,
        market["r"],
        market["q"],
        base_params,
        40_000,
        100,
        seed=11,
    )
    emp = CorrelationEffects.leverage_effect_from_paths(S, V)
    assert abs(emp - base_params.rho) < 0.05


def test_bk_exact_lower_variance_bias_than_euler(market):
    """Broadie-Kaya samples V_T from the exact CIR law; its terminal variance
    matches theory more closely than full-truncation Euler, whose truncation
    distorts the variance (the mean is nearly unbiased for both)."""
    from models.heston import HestonParams
    from simulation.heston_euler import HestonEulerSimulator
    from simulation.heston_exact import BroadieKayaSimulator

    # Feller-violating set so Euler's truncation bias is pronounced.
    p = HestonParams(kappa=1.0, theta=0.04, xi=0.9, rho=-0.7, V0=0.04)
    T, r, q = market["T"], market["r"], market["q"]
    ekt = np.exp(-p.kappa * T)
    var_theory = (
        p.V0 * p.xi**2 / p.kappa * ekt * (1 - ekt)
        + p.theta * p.xi**2 / (2 * p.kappa) * (1 - ekt) ** 2
    )
    _, Vb = BroadieKayaSimulator().simulate(100, p.V0, T, r, q, p, 60_000, 1, seed=5)
    _, Ve = HestonEulerSimulator().simulate(100, p.V0, T, r, q, p, 60_000, 50, seed=5)
    bias_bk = abs(Vb[:, -1].var() - var_theory)
    bias_eu = abs(Ve[:, -1].var() - var_theory)
    assert bias_bk < bias_eu
