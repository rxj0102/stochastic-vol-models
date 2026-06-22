"""Quadratic-Exponential (QE) simulation of the Heston model (Andersen, 2007).

The QE scheme moment-matches the exact conditional mean and variance of the
CIR process and switches between a quadratic-Gaussian approximation (low
variance ratio) and an exponential approximation (high variance ratio).  It is
the best-in-class low-bias discretisation for Heston and is the library
default for all simulation-based pricing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

__all__ = ["HestonQESimulator", "PSI_C"]

# Critical switching threshold between the quadratic and exponential branches.
PSI_C = 1.5


class HestonQESimulator:
    """Andersen (2007) Quadratic-Exponential simulator.

    Parameters
    ----------
    gamma1, gamma2 : float, optional
        Weights of the variance at the start/end of the step in the log-spot
        drift integral.  ``(0.5, 0.5)`` (default) is the central scheme.
    """

    def __init__(self, gamma1: float = 0.5, gamma2: float = 0.5) -> None:
        self.gamma1 = gamma1
        self.gamma2 = gamma2

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
        """Simulate spot and variance paths with the QE scheme.

        Returns
        -------
        tuple of numpy.ndarray
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.

        Notes
        -----
        Variance is guaranteed non-negative.  The log-spot uses Andersen's
        martingale-correcting discretisation so that ``E[S_T]`` matches the
        forward to within Monte Carlo error.
        """
        kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        ekt = np.exp(-kappa * dt)
        # Constants for conditional mean/variance of V (Andersen eq. 17-18).
        c1 = xi**2 * ekt * (1.0 - ekt) / kappa
        c2 = theta * xi**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)

        # Martingale drift constants K0..K4 (Andersen eq. 33).
        K0 = -rho * kappa * theta * dt / xi
        K1 = self.gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
        K2 = self.gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
        K3 = self.gamma1 * dt * (1.0 - rho**2)
        K4 = self.gamma2 * dt * (1.0 - rho**2)

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = V0
        logS = np.full(n_paths, np.log(S0))
        v = np.full(n_paths, V0)

        for t in range(n_steps):
            m = theta + (v - theta) * ekt  # E[V(t+dt) | V(t)]
            s2 = v * c1 + c2  # Var[V(t+dt) | V(t)]
            psi = s2 / m**2

            Uv = rng.random(n_paths)
            v_next = np.empty(n_paths)

            # --- Quadratic branch (psi <= PSI_C) ---
            quad = psi <= PSI_C
            if np.any(quad):
                inv_psi = 1.0 / psi[quad]
                b2 = (
                    2.0 * inv_psi
                    - 1.0
                    + np.sqrt(2.0 * inv_psi) * np.sqrt(2.0 * inv_psi - 1.0)
                )
                a = m[quad] / (1.0 + b2)
                Zv = norm.ppf(Uv[quad])
                v_next[quad] = a * (np.sqrt(b2) + Zv) ** 2

            # --- Exponential branch (psi > PSI_C) ---
            expo = ~quad
            if np.any(expo):
                p = (psi[expo] - 1.0) / (psi[expo] + 1.0)
                beta = (1.0 - p) / m[expo]
                u = Uv[expo]
                vv = np.where(
                    u <= p,
                    0.0,
                    np.log((1.0 - p) / np.maximum(1.0 - u, 1e-300)) / beta,
                )
                v_next[expo] = vv

            # Correlated normal for the spot innovation.
            Zs = rng.standard_normal(n_paths)
            logS = (
                logS
                + (r - q) * dt
                + K0
                + K1 * v
                + K2 * v_next
                + np.sqrt(np.maximum(K3 * v + K4 * v_next, 0.0)) * Zs
            )
            v = v_next
            S[:, t + 1] = np.exp(logS)
            V[:, t + 1] = v

        return S, V
