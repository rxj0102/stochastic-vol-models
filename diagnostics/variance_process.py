"""Analysis of the CIR variance process underlying Heston."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

__all__ = ["VarianceProcessDiagnostics"]


class VarianceProcessDiagnostics:
    """Closed-form properties of the CIR variance process.

    Parameters
    ----------
    params : object
        Object with ``kappa, theta, xi`` attributes.
    """

    def __init__(self, params) -> None:
        self.kappa = params.kappa
        self.theta = params.theta
        self.xi = params.xi

    # ------------------------------------------------------------------
    def stationary_distribution(self, v_grid: np.ndarray) -> np.ndarray:
        r"""Stationary Gamma density of the variance.

        The invariant law is :math:`V_\infty \sim
        \mathrm{Gamma}(2\kappa\theta/\xi^2,\ \xi^2/(2\kappa))` (shape, scale).
        """
        shape = 2.0 * self.kappa * self.theta / self.xi**2
        scale = self.xi**2 / (2.0 * self.kappa)
        return gamma_dist.pdf(v_grid, a=shape, scale=scale)

    def conditional_moments(self, V0: float, t_grid: np.ndarray) -> pd.DataFrame:
        r"""Conditional mean and variance of ``V(t)`` given ``V(0)=V0``."""
        t = np.asarray(t_grid, dtype=float)
        ekt = np.exp(-self.kappa * t)
        mean = V0 * ekt + self.theta * (1.0 - ekt)
        var = (
            V0 * self.xi**2 / self.kappa * ekt * (1.0 - ekt)
            + self.theta * self.xi**2 / (2.0 * self.kappa) * (1.0 - ekt) ** 2
        )
        return pd.DataFrame(
            {"t": t, "E_V": mean, "Var_V": var, "E_sqrtV": np.sqrt(np.maximum(mean, 0))}
        )

    def expected_realized_variance(self, V0: float, T: float) -> float:
        r"""Expected integrated variance :math:`E[\int_0^T V_t\,dt]`."""
        ekt = np.exp(-self.kappa * T)
        return float(self.theta * T + (V0 - self.theta) * (1.0 - ekt) / self.kappa)

    def variance_swap_rate(self, V0: float, T: float) -> float:
        """Fair variance-swap strike ``E[realized variance]/T`` (annualised)."""
        return self.expected_realized_variance(V0, T) / T

    def autocorrelation(self, lags: np.ndarray) -> np.ndarray:
        r"""Autocorrelation :math:`\mathrm{Corr}(V_t, V_{t+s}) = e^{-\kappa s}`."""
        return np.exp(-self.kappa * np.asarray(lags, dtype=float))

    def feller_summary(self) -> Dict[str, float]:
        """Return the Feller ratio and whether the boundary is unattainable."""
        ratio = 2.0 * self.kappa * self.theta / self.xi**2
        return {"feller_ratio": ratio, "feller_satisfied": ratio >= 1.0}
