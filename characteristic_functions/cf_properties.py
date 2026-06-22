"""Analysis utilities for the Heston characteristic function.

These helpers are used by the Streamlit *Fourier Pricing Lab* tab and by the
diagnostics layer to visualise and sanity-check the characteristic function.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .heston_cf import HestonCharacteristicFunction

__all__ = ["CFAnalyzer"]


class CFAnalyzer:
    """Diagnostics for the Heston characteristic function.

    Parameters
    ----------
    formulation : str, optional
        Forwarded to :class:`HestonCharacteristicFunction`.
    """

    def __init__(self, formulation: str = "little_trap") -> None:
        self.cf = HestonCharacteristicFunction(formulation)

    def cumulant_generating_function(
        self, u: np.ndarray, S0, V0, T, r, q, kappa, theta, xi, rho
    ) -> np.ndarray:
        r"""Return :math:`\ln \phi(-iu)` for real ``u`` (the CGF)."""
        vals = self.cf.mgf(u, S0, V0, T, r, q, kappa, theta, xi, rho)
        return np.log(vals)

    def cumulants(
        self, T, r, q, kappa, theta, xi, rho, V0
    ) -> Tuple[float, float, float, float]:
        """First four cumulants ``(c1, c2, c3, c4)`` of the log return.

        Returns
        -------
        tuple of float
            ``c1`` mean, ``c2`` variance, ``c3`` third cumulant,
            ``c4`` fourth cumulant.
        """
        c1 = HestonCharacteristicFunction._c1(T, kappa, theta, r, q, V0)
        c2 = HestonCharacteristicFunction._c2(T, kappa, theta, xi, rho, V0)
        c3 = HestonCharacteristicFunction._c3(T, kappa, theta, xi, rho, V0)
        c4 = HestonCharacteristicFunction._c4(T, kappa, theta, xi, rho, V0)
        return float(c1), float(c2), float(c3), float(c4)

    def cf_plot_data(
        self, u_range: np.ndarray, S0, V0, T, r, q, kappa, theta, xi, rho
    ) -> Dict[str, np.ndarray]:
        """Real/imaginary/modulus of the CF over ``u_range`` for plotting."""
        u = np.asarray(u_range, dtype=np.float64)
        vals = self.cf.cf(u, S0, V0, T, r, q, kappa, theta, xi, rho)
        return {
            "u": u,
            "real": vals.real,
            "imag": vals.imag,
            "abs": np.abs(vals),
        }

    def cf_convergence(
        self, u_max_range, S0, V0, K, T, r, q, kappa, theta, xi, rho
    ) -> Dict[str, np.ndarray]:
        """Lewis-style price as a function of the integration truncation.

        For each ``u_max`` the call price is computed by trapezoidal
        integration of the Lewis integrand; the sequence should converge as
        ``u_max`` grows, illustrating how far out the CF must be integrated.
        """
        u_max_range = np.asarray(u_max_range, dtype=np.float64)
        prices = np.empty_like(u_max_range)
        F = S0 * np.exp((r - q) * T)
        k = np.log(F / K)
        for i, umax in enumerate(u_max_range):
            u = np.linspace(1e-8, umax, 2000)
            phi = self.cf.cf(u - 0.5j, S0, V0, T, r, q, kappa, theta, xi, rho)
            integrand = (np.exp(1j * u * k) * phi / (u**2 + 0.25)).real
            integral = np.trapezoid(integrand, u)
            prices[i] = (
                S0 * np.exp(-q * T)
                - np.sqrt(S0 * K) * np.exp(-(r + q) * T / 2.0) / np.pi * integral
            )
        return {"u_max": u_max_range, "price": prices}

    def cf_stability_check(
        self, S0, V0, T, r, q, kappa, theta, xi, rho
    ) -> Dict[str, object]:
        """Flag numerical issues in the CF at extreme frequencies.

        Returns
        -------
        dict
            ``cf_at_zero`` (should be ~1), ``max_abs`` (should be <= 1 + tol),
            ``has_nan``, ``has_inf`` and an overall ``ok`` boolean.
        """
        u = np.linspace(-500, 500, 20001)
        vals = self.cf.cf(u, S0, V0, T, r, q, kappa, theta, xi, rho)
        cf0 = self.cf.cf(np.array([0.0]), S0, V0, T, r, q, kappa, theta, xi, rho)[0]
        absvals = np.abs(vals)
        return {
            "cf_at_zero": complex(cf0),
            "max_abs": float(np.nanmax(absvals)),
            "has_nan": bool(np.isnan(absvals).any()),
            "has_inf": bool(np.isinf(absvals).any()),
            "ok": bool(
                abs(cf0 - 1.0) < 1e-8
                and np.nanmax(absvals) < 1.0 + 1e-6
                and not np.isnan(absvals).any()
            ),
        }
