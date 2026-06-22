"""COS method for European option pricing (Fang & Oosterlee, 2008).

The COS method expands the (discounted) payoff in a Fourier-cosine series on a
truncated log-return interval ``[a, b]`` derived from the model cumulants.  It
converges *exponentially* in the number of terms ``N`` for smooth densities,
making it the fastest accurate pricer for the Heston model and the default
engine for calibration.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from characteristic_functions.heston_cf import HestonCharacteristicFunction

__all__ = ["COSMethodPricer", "DEFAULT_N_COS"]

DEFAULT_N_COS = 128
_DEFAULT_L = 12.0


class COSMethodPricer:
    """Price European vanillas under Heston via the COS method.

    Parameters
    ----------
    formulation : str, optional
        Forwarded to :class:`HestonCharacteristicFunction`.
    """

    def __init__(self, formulation: str = "little_trap") -> None:
        self.cf = HestonCharacteristicFunction(formulation)

    # ------------------------------------------------------------------
    @staticmethod
    def _chi_psi(k_idx, a, b, c, d):
        r"""Cosine-series coefficients :math:`\chi_k` and :math:`\psi_k`.

        These integrate :math:`e^y` and :math:`1` respectively against the
        cosine basis on :math:`[c, d]\subseteq[a, b]` (Fang-Oosterlee 2008).
        """
        bma = b - a
        omega = k_idx * np.pi / bma

        chi = (1.0 / (1.0 + omega**2)) * (
            np.cos(omega * (d - a)) * np.exp(d)
            - np.cos(omega * (c - a)) * np.exp(c)
            + omega * np.sin(omega * (d - a)) * np.exp(d)
            - omega * np.sin(omega * (c - a)) * np.exp(c)
        )

        psi = np.empty_like(k_idx, dtype=np.float64)
        psi[0] = d - c
        nz = k_idx[1:]
        omega_nz = nz * np.pi / bma
        psi[1:] = (np.sin(omega_nz * (d - a)) - np.sin(omega_nz * (c - a))) / omega_nz
        return chi, psi

    def _payoff_coeffs(self, k_idx, a, b, option_type):
        """Vanilla payoff cosine coefficients ``V_k`` (unit strike)."""
        if option_type == "call":
            chi, psi = self._chi_psi(k_idx, a, b, 0.0, b)
            return (2.0 / (b - a)) * (chi - psi)
        # put
        chi, psi = self._chi_psi(k_idx, a, b, a, 0.0)
        return (2.0 / (b - a)) * (-chi + psi)

    def price(
        self,
        S0,
        K,
        T,
        r,
        q,
        params,
        option_type: str = "call",
        N: int = DEFAULT_N_COS,
        L: float = _DEFAULT_L,
    ) -> float:
        r"""Price a single European option.

        Parameters
        ----------
        S0, K, T, r, q : float
            Contract data.
        params : object
            Object with attributes ``kappa, theta, xi, rho, V0``.
        option_type : {'call', 'put'}, optional
        N : int, optional
            Number of cosine terms.
        L : float, optional
            Truncation-range multiplier for the cumulant interval.

        Returns
        -------
        float
            Option premium.
        """
        kappa, theta, xi, rho, V0 = (
            params.kappa,
            params.theta,
            params.xi,
            params.rho,
            params.V0,
        )
        # Cumulants of x = ln(S_T / K).  c1 = ln(S0/K) + mean log return.
        c1 = np.log(S0 / K) + HestonCharacteristicFunction._c1(
            T, kappa, theta, r, q, V0
        )
        c2 = HestonCharacteristicFunction._c2(T, kappa, theta, xi, rho, V0)
        c4 = HestonCharacteristicFunction._c4(T, kappa, theta, xi, rho, V0)
        c2 = max(c2, 1e-10)

        a = c1 - L * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))
        b = c1 + L * np.sqrt(c2 + np.sqrt(max(c4, 0.0)))

        k_idx = np.arange(N)
        omega = k_idx * np.pi / (b - a)

        # Characteristic function of x = ln(S_T) - ln K, evaluated at omega.
        # phi_x(u) = E[e^{i u (ln S_T - ln K)}] = cf(u) * e^{-i u ln K}.
        phi = self.cf.cf(omega, S0, V0, T, r, q, kappa, theta, xi, rho)
        phi = phi * np.exp(-1j * omega * np.log(K))

        Vk = self._payoff_coeffs(k_idx, a, b, option_type)
        Fk = (phi * np.exp(-1j * omega * a)).real
        Fk[0] *= 0.5  # prime on the sum

        price = K * np.exp(-r * T) * np.sum(Fk * Vk)
        return float(price)

    def price_surface(
        self, S0, strikes, maturities, r, q, params, option_type="call", N=DEFAULT_N_COS
    ) -> np.ndarray:
        """Price a grid of options, shape ``(len(maturities), len(strikes))``."""
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        out = np.empty((maturities.size, strikes.size))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                out[i, j] = self.price(S0, K, T, r, q, params, option_type, N=N)
        return out

    def convergence_test(
        self,
        S0,
        K,
        T,
        r,
        q,
        params,
        option_type="call",
        N_range: List[int] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Price for a sequence of ``N`` to demonstrate exponential convergence."""
        if N_range is None:
            N_range = [8, 16, 32, 64, 128, 256, 512]
        prices = np.array(
            [self.price(S0, K, T, r, q, params, option_type, N=n) for n in N_range]
        )
        ref = self.price(S0, K, T, r, q, params, option_type, N=1024)
        return {
            "N": np.asarray(N_range),
            "price": prices,
            "abs_error": np.abs(prices - ref),
            "reference": ref,
        }
