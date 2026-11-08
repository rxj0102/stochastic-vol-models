import warnings

import numpy as np

from models import HestonModel, HestonParams


def test_simulate_respects_rate_and_dividend():
    warnings.simplefilter("ignore")
    m = HestonModel(HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, V0=0.04))
    S, _ = m.simulate(100.0, 0.04, 1.0, 40000, 100, seed=1, r=0.05, q=0.0)
    assert abs(S[:, -1].mean() / (100 * np.exp(0.05)) - 1) < 0.01
    S0, _ = m.simulate(100.0, 0.04, 1.0, 40000, 100, seed=1)
    assert abs(S0[:, -1].mean() / 100 - 1) < 0.01
