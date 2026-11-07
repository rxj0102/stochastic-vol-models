"""Euler-Maruyama simulation of the Heston model (full-truncation scheme).

Uses log-Euler for the spot (guaranteeing positivity) and a full-truncation
Euler step for the variance (Lord, Koekkoek & van Dijk, 2010), which is the
most robust of the simple truncation fixes for the CIR process.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["HestonEulerSimulator"]


class HestonEulerSimulator:
    """Vectorised Euler-Maruyama simulator for the Heston SDE system."""

    def simulate(
        self,
        S0: float,
        V0: float,
        T: float,
        r: float,
        q: float,
        params,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
        antithetic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate spot and variance paths.

        Parameters
        ----------
        S0, V0, T, r, q : float
            Spot, initial variance, horizon, rate, dividend yield.
        params : object
            Object with ``kappa, theta, xi, rho`` attributes.
        n_paths, n_steps : int
            Number of Monte Carlo paths and time steps.
        seed : int, optional
            RNG seed.
        antithetic : bool, optional
            If ``True``, the second half of the paths uses the negated normals
            of the first half for variance reduction (``n_paths`` is rounded up
            to an even number).

        Returns
        -------
        tuple of numpy.ndarray
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.
        """
        kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        if antithetic:
            half = (n_paths + 1) // 2
            Z1h = rng.standard_normal((half, n_steps))
            Z2h = rng.standard_normal((half, n_steps))
            Z1 = np.vstack([Z1h, -Z1h])[:n_paths]
            Z2 = np.vstack([Z2h, -Z2h])[:n_paths]
        else:
            Z1 = rng.standard_normal((n_paths, n_steps))
            Z2 = rng.standard_normal((n_paths, n_steps))

        dW_S = sqrt_dt * Z1
        dW_V = sqrt_dt * (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2)

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = V0

        logS = np.full(n_paths, np.log(S0))
        v = np.full(n_paths, V0)
        for t in range(n_steps):
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            logS = logS + (r - q - 0.5 * v_pos) * dt + sqrt_v * dW_S[:, t]
            v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * dW_V[:, t]
            S[:, t + 1] = np.exp(logS)
            V[:, t + 1] = v
        # Report the truncated variance (paths never used negative vol in S).
        V = np.maximum(V, 0.0)
        return S, V
