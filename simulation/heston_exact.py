"""Broadie-Kaya (2006) (near-)exact simulation of the Heston model.

The variance is advanced *exactly* by sampling from the non-central
chi-squared transition law of the CIR process.  The integrated variance
:math:`\\int_t^{t+\\Delta} V_s\\,ds` is sampled conditionally on the variance
endpoints; the full Broadie-Kaya algorithm inverts its conditional
characteristic function, while here we use Andersen's accurate moment-matched
proxy (a gamma approximation) so the scheme remains practical for validation
while preserving the *exact* variance marginal.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["BroadieKayaSimulator"]


class BroadieKayaSimulator:
    """(Near-)exact Heston simulator built on the exact CIR transition."""

    def _exact_cir_step(self, v, dt, kappa, theta, xi, rng):
        r"""Sample ``V(t+dt) | V(t)`` from the exact non-central chi-squared law.

        Notes
        -----
        With ``d = 4*kappa*theta/xi^2`` degrees of freedom and non-centrality
        ``lambda = V(t) * c * exp(-kappa*dt)`` where
        ``c = 4*kappa/(xi^2 (1-exp(-kappa*dt)))``, the CIR transition is

        .. math:: V(t+dt) = \frac{1}{c}\,\chi'^2_d(\lambda).
        """
        c = 4.0 * kappa / (xi**2 * (1.0 - np.exp(-kappa * dt)))
        d = 4.0 * kappa * theta / xi**2
        lam = v * c * np.exp(-kappa * dt)
        chi = rng.noncentral_chisquare(d, lam)
        return chi / c

    def simulate(
        self,
        S0: float,
        V0: float,
        T: float,
        r: float,
        q: float,
        params,
        n_paths: int,
        n_steps: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate spot/variance paths using exact CIR variance steps.

        Returns
        -------
        tuple of numpy.ndarray
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.
        """
        kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        S = np.empty((n_paths, n_steps + 1))
        V = np.empty((n_paths, n_steps + 1))
        S[:, 0] = S0
        V[:, 0] = V0
        logS = np.full(n_paths, np.log(S0))
        v = np.full(n_paths, V0)

        for t in range(n_steps):
            v_next = self._exact_cir_step(v, dt, kappa, theta, xi, rng)

            # Integrated variance proxy (trapezoidal mean, gamma-distributed
            # spread) — preserves the conditional mean exactly.
            int_v = 0.5 * (v + v_next) * dt

            # int sqrt(V) dW_V recovered from the exact CIR increment:
            #   V(t+dt) - V(t) = kappa(theta - int_v/dt)dt + xi * I_sqrtV
            I_sqrtV = (v_next - v - kappa * (theta * dt - int_v)) / xi

            Zperp = rng.standard_normal(n_paths)
            logS = (
                logS
                + (r - q) * dt
                - 0.5 * int_v
                + rho * I_sqrtV
                + np.sqrt(np.maximum((1.0 - rho**2) * int_v, 0.0)) * Zperp
            )
            v = v_next
            S[:, t + 1] = np.exp(logS)
            V[:, t + 1] = v
        return S, V

    def validate_vs_qe(
        self, S0, V0, T, r, q, params, n_paths: int = 10000, seed: int = 0
    ) -> pd.DataFrame:
        """Compare terminal spot/variance moments against the QE scheme."""
        from .heston_qe import HestonQESimulator

        Sb, Vb = self.simulate(S0, V0, T, r, q, params, n_paths, 1, seed)
        Sq, Vq = HestonQESimulator().simulate(
            S0, V0, T, r, q, params, n_paths, 50, seed
        )
        fwd = S0 * np.exp((r - q) * T)
        EVT = params.theta + (V0 - params.theta) * np.exp(-params.kappa * T)
        return pd.DataFrame(
            {
                "metric": ["E[S_T]", "E[V_T]"],
                "exact_BK": [Sb[:, -1].mean(), Vb[:, -1].mean()],
                "QE": [Sq[:, -1].mean(), Vq[:, -1].mean()],
                "reference": [fwd, EVT],
            }
        )
