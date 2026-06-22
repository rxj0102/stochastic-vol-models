"""Greek-based daily P&L attribution for Heston positions."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .greeks_heston import HestonGreeks

__all__ = ["VolPnLAttribution"]


class VolPnLAttribution:
    """Decompose a position's P&L into Greek contributions.

    The Heston-specific cross terms (vanna, volga) are included alongside the
    classic delta/gamma/vega/theta decomposition.
    """

    def __init__(self) -> None:
        self.greeks = HestonGreeks()

    def explain(
        self,
        S0,
        K,
        T,
        r,
        q,
        params,
        dS,
        dSigma,
        dt=1.0 / 252,
        dr=0.0,
        option_type="call",
    ) -> Dict[str, float]:
        r"""Attribute the P&L of a one-step market move.

        Parameters
        ----------
        dS : float
            Change in spot.
        dSigma : float
            Change in ``sigma0 = sqrt(V0)`` (vol points).
        dt : float
            Time step in years.
        dr : float
            Change in the risk-free rate.

        Returns
        -------
        dict
            Per-Greek P&L contributions plus ``predicted``, ``actual`` and
            ``residual``.
        """
        g = self.greeks.compute_all(S0, K, T, r, q, params, option_type)

        delta_pnl = g["delta"] * dS
        gamma_pnl = 0.5 * g["gamma"] * dS**2
        vega_pnl = g["vega"] * dSigma
        volga_pnl = 0.5 * g["volga"] * dSigma**2
        vanna_pnl = g["vanna"] * dS * dSigma
        theta_pnl = g["theta"] * dt
        rho_pnl = g["rho_r"] * dr

        predicted = (
            delta_pnl
            + gamma_pnl
            + vega_pnl
            + volga_pnl
            + vanna_pnl
            + theta_pnl
            + rho_pnl
        )

        # Actual full reprice after the move.
        from dataclasses import replace

        sigma0 = np.sqrt(params.V0)
        new_params = replace(params, V0=max((sigma0 + dSigma) ** 2, 1e-8))
        actual = (
            self.greeks._price(
                S0 + dS, K, max(T - dt, 1e-6), r + dr, q, new_params, option_type
            )
            - g["price"]
        )
        return {
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "volga_pnl": volga_pnl,
            "vanna_pnl": vanna_pnl,
            "theta_pnl": theta_pnl,
            "rho_pnl": rho_pnl,
            "predicted": predicted,
            "actual": actual,
            "residual": actual - predicted,
        }

    def waterfall_data(self, explain_result: Dict[str, float]) -> Dict[str, list]:
        """Reshape an :meth:`explain` result into Plotly waterfall arrays."""
        keys = [
            "delta_pnl",
            "gamma_pnl",
            "vega_pnl",
            "vanna_pnl",
            "volga_pnl",
            "theta_pnl",
            "rho_pnl",
            "residual",
        ]
        labels = [
            "Delta",
            "Gamma",
            "Vega",
            "Vanna",
            "Volga",
            "Theta",
            "Rho",
            "Residual",
        ]
        values = [explain_result[k] for k in keys]
        return {"labels": labels, "values": values, "total": explain_result["actual"]}
