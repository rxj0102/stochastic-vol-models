"""Heston Greeks via finite differencing of the COS pricer.

Closed-form CF-differentiated Greeks exist, but bumping the (fast, smooth) COS
price is accurate to many digits and far less error-prone.  Spot Greeks use
the spot, vol Greeks use ``sigma0 = sqrt(V0)`` so that vega/volga are quoted in
the market-standard "per vol point" units.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from fourier_pricing.cos_method import COSMethodPricer
from models.heston import HestonParams

__all__ = ["HestonGreeks"]


class HestonGreeks:
    """Semi-analytical (bump-and-reprice) Greeks under the Heston model."""

    def __init__(self) -> None:
        self.pricer = COSMethodPricer()

    def _price(self, S0, K, T, r, q, params, option_type):
        return self.pricer.price(S0, K, T, r, q, params, option_type)

    def _with_V0(self, params: HestonParams, V0: float) -> HestonParams:
        return replace(params, V0=V0)

    def compute_all(
        self, S0, K, T, r, q, params: HestonParams, option_type="call"
    ) -> Dict[str, float]:
        """Return all first- and second-order Greeks as a dict.

        Keys: ``price, delta, gamma, vega, volga, vanna, theta, rho_r,
        rho_corr``.  ``vega``/``volga`` are per unit of ``sigma0 = sqrt(V0)``;
        ``rho_corr`` is sensitivity to the Heston correlation parameter.
        """
        p = params
        hS = 1e-3 * S0
        sigma0 = np.sqrt(p.V0)
        hsig = 1e-3

        base = self._price(S0, K, T, r, q, p, option_type)

        # Spot Greeks.
        up = self._price(S0 + hS, K, T, r, q, p, option_type)
        dn = self._price(S0 - hS, K, T, r, q, p, option_type)
        delta = (up - dn) / (2 * hS)
        gamma = (up - 2 * base + dn) / hS**2

        # Vol Greeks (bump sigma0 -> V0 = sigma0^2).
        p_up = self._with_V0(p, (sigma0 + hsig) ** 2)
        p_dn = self._with_V0(p, max((sigma0 - hsig) ** 2, 1e-8))
        v_up = self._price(S0, K, T, r, q, p_up, option_type)
        v_dn = self._price(S0, K, T, r, q, p_dn, option_type)
        vega = (v_up - v_dn) / (2 * hsig)
        volga = (v_up - 2 * base + v_dn) / hsig**2

        # Vanna: d^2 Price / (dS0 d sigma0).
        v_up_Sup = self._price(S0 + hS, K, T, r, q, p_up, option_type)
        v_up_Sdn = self._price(S0 - hS, K, T, r, q, p_up, option_type)
        v_dn_Sup = self._price(S0 + hS, K, T, r, q, p_dn, option_type)
        v_dn_Sdn = self._price(S0 - hS, K, T, r, q, p_dn, option_type)
        vanna = ((v_up_Sup - v_up_Sdn) - (v_dn_Sup - v_dn_Sdn)) / (4 * hS * hsig)

        # Theta (per year; price decay as maturity shrinks).
        hT = min(1e-3, 0.5 * T)
        theta = -(
            self._price(S0, K, T + hT, r, q, p, option_type)
            - self._price(S0, K, T - hT, r, q, p, option_type)
        ) / (2 * hT)

        # Rate rho.
        hr = 1e-4
        rho_r = (
            self._price(S0, K, T, r + hr, q, p, option_type)
            - self._price(S0, K, T, r - hr, q, p, option_type)
        ) / (2 * hr)

        # Correlation sensitivity (Heston-specific).
        hc = 1e-3
        p_rc_up = replace(p, rho=min(p.rho + hc, 0.999))
        p_rc_dn = replace(p, rho=max(p.rho - hc, -0.999))
        rho_corr = (
            self._price(S0, K, T, r, q, p_rc_up, option_type)
            - self._price(S0, K, T, r, q, p_rc_dn, option_type)
        ) / (2 * hc)

        return {
            "price": base,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "volga": volga,
            "vanna": vanna,
            "theta": theta,
            "rho_r": rho_r,
            "rho_corr": rho_corr,
        }
