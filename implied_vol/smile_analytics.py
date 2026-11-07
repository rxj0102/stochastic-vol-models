"""Smile analytics: ATM level, skew, curvature and term structure."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

from .surface import IVSurface

__all__ = ["SmileAnalytics"]


class SmileAnalytics:
    """Compute standard smile metrics from an :class:`IVSurface`."""

    def _strike_for_delta(self, delta, sig, F, T, r, q, is_call=True):
        r"""Invert Black-Scholes delta to a strike (forward delta convention)."""
        # delta = e^{-qT} N(d1) for a call.  Solve for K.
        sign = 1.0 if is_call else -1.0
        n_inv = (
            norm.ppf(delta * np.exp(q * T))
            if is_call
            else norm.ppf(1.0 - delta * np.exp(q * T))
        )
        d1 = sign * n_inv
        # d1 = (ln(F/K) + 0.5 sig^2 T)/(sig sqrt(T))  =>  ln(F/K) = d1 sig sqrt T - 0.5 sig^2 T
        lnFK = d1 * sig * np.sqrt(T) - 0.5 * sig**2 * T
        return F * np.exp(-lnFK)

    def compute_smile_metrics(self, iv_surface: IVSurface, S0, r, q) -> pd.DataFrame:
        """Return a per-maturity DataFrame of smile metrics.

        Columns: ``maturity, atm_vol, skew_25d, butterfly_25d, rr_25d,
        put_call_skew``.
        """
        rows = []
        for i, T in enumerate(iv_surface.maturities):
            F = S0 * np.exp((r - q) * T)
            atm = iv_surface.iv_at(F, T)
            # 25-delta strikes (approx via ATM vol).
            k25c = self._strike_for_delta(0.25, atm, F, T, r, q, is_call=True)
            k25p = self._strike_for_delta(0.25, atm, F, T, r, q, is_call=False)
            iv_25c = iv_surface.iv_at(k25c, T)
            iv_25p = iv_surface.iv_at(k25p, T)
            rr = iv_25c - iv_25p  # 25-delta risk reversal
            bf = 0.5 * (iv_25c + iv_25p) - atm  # 25-delta butterfly
            # Local skew d(IV)/d ln K at the forward (finite difference).
            dk = 0.01 * F
            skew = (iv_surface.iv_at(F + dk, T) - iv_surface.iv_at(F - dk, T)) / (
                2.0 * dk / F
            )
            rows.append(
                {
                    "maturity": float(T),
                    "atm_vol": atm,
                    "skew_25d": rr,
                    "butterfly_25d": bf,
                    "rr_25d": rr,
                    "put_call_skew": iv_25p - iv_25c,
                    "local_skew": skew,
                }
            )
        return pd.DataFrame(rows)

    def term_structure_of_vol(
        self, iv_surface: IVSurface, S0, r, q
    ) -> Dict[str, np.ndarray]:
        """ATM vol as a function of maturity."""
        mats = iv_surface.maturities
        atm = np.array([iv_surface.iv_at(S0 * np.exp((r - q) * T), T) for T in mats])
        return {"maturity": mats, "atm_vol": atm}

    def skew_decomposition(self, params, maturities) -> pd.DataFrame:
        r"""Attribute smile features to Heston parameters.

        Heuristic decomposition used for exposition:

        * short-dated skew scales with :math:`\rho \xi`,
        * curvature scales with :math:`\xi^2 (1-\rho^2)`,
        * the skew decay term-structure scales with :math:`1/\kappa`.
        """
        rows = []
        for T in np.asarray(maturities, dtype=float):
            decay = (1.0 - np.exp(-params.kappa * T)) / (params.kappa * T)
            rows.append(
                {
                    "maturity": T,
                    "skew_proxy": params.rho * params.xi * decay,
                    "curvature_proxy": params.xi**2 * (1.0 - params.rho**2) * decay,
                    "skew_decay": decay,
                }
            )
        return pd.DataFrame(rows)
