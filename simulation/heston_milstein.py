"""Milstein-scheme simulation of the Heston variance process.

The Milstein correction adds the second-order term ``0.25*xi^2*(dW^2 - dt)`` to
the CIR Euler step, reducing the discretisation bias in the variance compared
to plain Euler.  Full truncation keeps the variance non-negative.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["HestonMilsteinSimulator"]


class HestonMilsteinSimulator:
    """Milstein simulator for the Heston SDE system."""

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate spot/variance paths with the Milstein variance scheme.

        Returns
        -------
        tuple of numpy.ndarray
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.
        """
        kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

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
            v = (
                v
                + kappa * (theta - v_pos) * dt
                + xi * sqrt_v * dW_V[:, t]
                + 0.25 * xi**2 * (dW_V[:, t] ** 2 - dt)
            )
            S[:, t + 1] = np.exp(logS)
            V[:, t + 1] = np.maximum(v, 0.0)
        return S, np.maximum(V, 0.0)
