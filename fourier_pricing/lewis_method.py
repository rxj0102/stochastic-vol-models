"""Lewis (2001) Fourier option pricing.

Lewis derived a single-integral representation of the European call along the
contour :math:`\\operatorname{Im}(u) = -1/2`, where the integrand decays
quickly and the integral can be evaluated with adaptive quadrature.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from characteristic_functions.heston_cf import HestonCharacteristicFunction

__all__ = ["LewisMethodPricer"]


class LewisMethodPricer:
    """Lewis (2001) semi-analytical pricer for Heston European options.

    Parameters
    ----------
    formulation : str, optional
        Forwarded to :class:`HestonCharacteristicFunction`.
    """

    def __init__(self, formulation: str = "little_trap") -> None:
        self.cf = HestonCharacteristicFunction(formulation)

    def price(
        self,
        S0,
        K,
        T,
        r,
        q,
        params,
        option_type: str = "call",
        N_quad: int = 500,
    ) -> float:
        r"""Price a single European option.

        Notes
        -----
        With :math:`k = \ln(F/K)` and :math:`F = S_0 e^{(r-q)T}`,

        .. math::

            C = S_0 e^{-qT} - \frac{\sqrt{S_0 K}\, e^{-(r+q)T/2}}{\pi}
                \int_0^\infty \operatorname{Re}\!\left[
                e^{i u k}\,\frac{\phi(u - i/2)}{u^2 + 1/4}\right] du.
        """
        kappa, theta, xi, rho, V0 = (
            params.kappa,
            params.theta,
            params.xi,
            params.rho,
            params.V0,
        )
        F = S0 * np.exp((r - q) * T)
        k = np.log(F / K)
        lnF = np.log(F)

        def integrand(u: float) -> float:
            # Forward-centered CF: phi_F(z) = phi(z) * exp(-i z ln F).
            z = u - 0.5j
            phi = self.cf.cf(np.array([z]), S0, V0, T, r, q, kappa, theta, xi, rho)[0]
            phi_F = phi * np.exp(-1j * z * lnF)
            return float((np.exp(1j * u * k) * phi_F / (u**2 + 0.25)).real)

        integral, _ = quad(integrand, 0.0, np.inf, limit=N_quad)
        call = (
            S0 * np.exp(-q * T)
            - np.sqrt(S0 * K) * np.exp(-(r + q) * T / 2.0) / np.pi * integral
        )
        call = max(call, 0.0)
        if option_type == "call":
            return call
        return max(call - S0 * np.exp(-q * T) + K * np.exp(-r * T), 0.0)

    def price_batch(
        self, S0, strikes, T, r, q, params, option_type="call"
    ) -> np.ndarray:
        """Vectorised wrapper pricing each strike in ``strikes``."""
        strikes = np.asarray(strikes, dtype=np.float64)
        return np.array(
            [self.price(S0, float(K), T, r, q, params, option_type) for K in strikes]
        )
