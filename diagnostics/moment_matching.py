"""Compare theoretical Heston log-return moments to Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from characteristic_functions.heston_cf import HestonCharacteristicFunction
from simulation.heston_qe import HestonQESimulator
from utils.statistics import moments_from_samples

__all__ = ["MomentMatching"]


class MomentMatching:
    """Validate the model's analytical moments against simulated paths."""

    def __init__(self) -> None:
        self.cf = HestonCharacteristicFunction()
        self.sim = HestonQESimulator()

    def theoretical_moments(self, params, S0, T, r, q):
        """Return ``(mean, var, skew, kurt)`` of the log return from the CF."""
        m = self.cf.log_return_moments(
            T, r, q, params.kappa, params.theta, params.xi, params.rho, params.V0
        )
        return m["mean"], m["variance"], m["skewness"], m["kurtosis"]

    def simulated_moments(self, S_paths, S0):
        """Return ``(mean, var, skew, kurt)`` of the simulated log return."""
        logret = np.log(S_paths[:, -1] / S0)
        m = moments_from_samples(logret)
        return m["mean"], m["variance"], m["skewness"], m["kurtosis"]

    def comparison_table(
        self, params, S0, T, r, q, n_paths=100_000, n_steps=100, seed=0
    ) -> pd.DataFrame:
        """Theory-vs-simulation table for the first four moments."""
        tm = self.theoretical_moments(params, S0, T, r, q)
        S, _ = self.sim.simulate(S0, params.V0, T, r, q, params, n_paths, n_steps, seed)
        sm = self.simulated_moments(S, S0)
        names = ["mean", "variance", "skewness", "excess_kurtosis"]
        return pd.DataFrame(
            {
                "moment": names,
                "theoretical": tm,
                "simulated": sm,
                "abs_diff": np.abs(np.array(tm) - np.array(sm)),
            }
        )
