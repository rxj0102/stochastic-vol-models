"""Damping / regularisation analysis for Carr-Madan Fourier inversion.

The Carr-Madan method multiplies the call payoff by :math:`e^{\\alpha k}` so
that the modified payoff is integrable.  This module helps choose and validate
the damping parameter :math:`\\alpha`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .heston_cf import HestonCharacteristicFunction

__all__ = ["DampeningAnalyzer", "DEFAULT_ALPHA_CM"]

DEFAULT_ALPHA_CM = 1.5


class DampeningAnalyzer:
    """Choose and diagnose the Carr-Madan damping parameter ``alpha``.

    Parameters
    ----------
    formulation : str, optional
        Forwarded to :class:`HestonCharacteristicFunction`.
    """

    def __init__(self, formulation: str = "little_trap") -> None:
        self.cf = HestonCharacteristicFunction(formulation)

    def _psi(self, v, alpha, S0, V0, T, r, q, kappa, theta, xi, rho):
        """Carr-Madan integrand numerator ``psi(v)`` (without the FFT phase)."""
        u = v - (alpha + 1.0) * 1j
        phi = self.cf.cf(u, S0, V0, T, r, q, kappa, theta, xi, rho)
        denom = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
        return np.exp(-r * T) * phi / denom

    def optimal_alpha(
        self,
        S0,
        K,
        T,
        r,
        params,
        q=0.0,
        method: str = "automatic",
        alpha_grid: np.ndarray | None = None,
    ) -> float:
        r"""Return a damping parameter for a call.

        Parameters
        ----------
        S0, K, T, r : float
            Contract data.
        params : object
            Object exposing ``kappa, theta, xi, rho, V0`` attributes.
        q : float, optional
            Dividend yield.
        method : {'automatic', 'optimal'}, optional
            ``'automatic'`` returns :data:`DEFAULT_ALPHA_CM`.  ``'optimal'``
            minimises :math:`|\psi(0)|` over ``alpha_grid``, an approximation to
            the Lord-Kahl optimal-contour choice.

        Returns
        -------
        float
            The selected damping parameter (positive, for calls).
        """
        if method == "automatic":
            return DEFAULT_ALPHA_CM
        if alpha_grid is None:
            alpha_grid = np.linspace(0.05, 3.0, 60)
        norms = []
        for a in alpha_grid:
            val = self._psi(
                0.0,
                a,
                S0,
                params.V0,
                T,
                r,
                q,
                params.kappa,
                params.theta,
                params.xi,
                params.rho,
            )
            norms.append(np.abs(val))
        return float(alpha_grid[int(np.argmin(norms))])

    def alpha_sensitivity(
        self, S0, K, T, r, params, alpha_range, q=0.0
    ) -> Dict[str, np.ndarray]:
        """Call price as a function of the damping parameter.

        A robust pricing method should return (nearly) the same price for any
        ``alpha`` in the admissible range; this produces the data to show that.
        """
        from fourier_pricing.carr_madan import CarrMadanPricer

        pricer = CarrMadanPricer(formulation=self.cf.formulation)
        alpha_range = np.asarray(alpha_range, dtype=np.float64)
        prices = np.array(
            [
                pricer.price(S0, K, T, r, q, params, "call", alpha=float(a))
                for a in alpha_range
            ]
        )
        return {"alpha": alpha_range, "price": prices}

    def truncation_error(self, u_max, S0, K, T, r, params, alpha, q=0.0) -> float:
        r"""Estimate the tail integral beyond ``u_max`` for the call.

        Returns
        -------
        float
            :math:`\frac{1}{\pi}\int_{u_{\max}}^{\infty} |\psi(v)|\,dv`
            approximated on a fine grid — an upper bound on the truncation
            error of the Carr-Madan integral.
        """
        v = np.linspace(u_max, u_max + 200.0, 4000)
        psi = self._psi(
            v,
            alpha,
            S0,
            params.V0,
            T,
            r,
            q,
            params.kappa,
            params.theta,
            params.xi,
            params.rho,
        )
        return float(np.trapezoid(np.abs(psi), v) / np.pi)
