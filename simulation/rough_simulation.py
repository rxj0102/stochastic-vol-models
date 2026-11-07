"""Rough Bergomi (rBergomi) simulation via the hybrid scheme sketch.

The variance is driven by a Volterra fractional Brownian motion with Hurst
exponent ``H < 1/2``.  We build the Riemann-Liouville fBM by exact Cholesky of
its covariance (accurate for the modest path counts used in the dashboard).
"""

from __future__ import annotations

from math import gamma
from typing import Optional, Tuple

import numpy as np

__all__ = ["RoughVolSimulator"]


class RoughVolSimulator:
    """rBergomi simulator using Cholesky-exact Volterra fBM."""

    @staticmethod
    def _volterra_covariance(t_grid: np.ndarray, H: float) -> np.ndarray:
        r"""Covariance of the Riemann-Liouville fBM :math:`W^H` on ``t_grid``.

        Uses the closed form (Bayer-Friz-Gatheral, 2016)

        .. math::

            \mathrm{Cov}(W^H_s, W^H_t) = \frac{s^{H+1/2} t^{H-1/2}}{H+1/2}
            \,{}_2F_1\!\left(1, \tfrac12 - H, H+\tfrac32, \tfrac{s}{t}\right),
            \quad s \le t,

        approximated here by direct numerical integration of the kernel for
        robustness without SciPy's hypergeometric function.
        """
        n = t_grid.size
        cov = np.zeros((n, n))
        # Fine grid for kernel integration.
        for i in range(n):
            for j in range(i, n):
                s, t = t_grid[i], t_grid[j]
                lo = min(s, t)
                if lo <= 0:
                    cov[i, j] = cov[j, i] = 0.0
                    continue
                u = np.linspace(1e-9, lo, 200)
                Ks = (s - u) ** (H - 0.5)
                Kt = (t - u) ** (H - 0.5)
                val = np.trapezoid(Ks * Kt, u) / (gamma(H + 0.5) ** 2)
                cov[i, j] = cov[j, i] = val
        return cov

    def simulate_rbergomi(
        self,
        S0: float,
        xi0: float,
        eta: float,
        H: float,
        rho: float,
        T: float,
        r: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate rBergomi spot and variance paths.

        Returns
        -------
        tuple of numpy.ndarray
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        t_grid = np.linspace(dt, T, n_steps)

        cov = self._volterra_covariance(t_grid, H)
        # Jitter for numerical positive-definiteness.
        cov += 1e-12 * np.eye(n_steps)
        L = np.linalg.cholesky(cov)

        Z = rng.standard_normal((n_steps, n_paths))
        WH = L @ Z  # (n_steps, n_paths) Volterra fBM

        # Variance: V_t = xi0 * exp(eta WH_t - 0.5 eta^2 t^{2H}).
        var_drift = 0.5 * eta**2 * t_grid ** (2.0 * H)
        V_inner = xi0 * np.exp(eta * WH - var_drift[:, None])

        V = np.empty((n_paths, n_steps + 1))
        V[:, 0] = xi0
        V[:, 1:] = V_inner.T

        # Spot driven by Brownian motion correlated with the fBM driver.
        Zs = rng.standard_normal((n_paths, n_steps))
        logS = np.full(n_paths, np.log(S0))
        S = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        dWperp = np.sqrt(dt) * Zs
        # Correlated Brownian increment for the spot.
        dWcorr = np.sqrt(dt) * (rho * Z.T + np.sqrt(1.0 - rho**2) * Zs)
        for t in range(n_steps):
            v = V[:, t]
            logS = (
                logS + (r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0.0)) * dWcorr[:, t]
            )
            S[:, t + 1] = np.exp(logS)
        return S, V
