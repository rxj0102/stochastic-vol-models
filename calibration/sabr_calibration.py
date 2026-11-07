"""Slice-by-slice SABR calibration (one parameter set per maturity)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.sabr import hagan_lognormal_vol

from .market_data import MarketIVSurface

__all__ = ["SABRCalibrator"]


class SABRCalibrator:
    """Calibrate SABR (alpha, rho, nu) per maturity, with beta fixed by default."""

    def calibrate_slice(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        F: float,
        T: float,
        beta: float = 0.5,
    ) -> dict:
        """Fit ``(alpha, rho, nu)`` to one maturity slice.

        Returns
        -------
        dict
            ``{alpha, beta, rho, nu, rmse}``.
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        market_ivs = np.asarray(market_ivs, dtype=np.float64)
        mask = np.isfinite(market_ivs)
        strikes, market_ivs = strikes[mask], market_ivs[mask]

        atm_idx = int(np.argmin(np.abs(strikes - F)))
        alpha0 = market_ivs[atm_idx] * F ** (1.0 - beta)

        def obj(p):
            alpha, rho, nu = p
            if alpha <= 0 or nu <= 0 or not -0.999 < rho < 0.999:
                return 1e6
            model = hagan_lognormal_vol(F, strikes, T, alpha, beta, rho, nu)
            return np.sqrt(np.mean((model - market_ivs) ** 2))

        res = minimize(
            obj,
            x0=[alpha0, -0.3, 0.5],
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
        )
        alpha, rho, nu = res.x
        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "rho": float(rho),
            "nu": float(nu),
            "rmse": float(res.fun),
        }

    def calibrate_surface(
        self,
        market_surface: MarketIVSurface,
        forwards: Optional[dict] = None,
        beta: float = 0.5,
    ) -> pd.DataFrame:
        """Calibrate every maturity slice; return a DataFrame of parameters."""
        rows = []
        for T in market_surface.maturities:
            sub = market_surface.data[market_surface.data["maturity"] == T]
            F = (
                forwards.get(T, market_surface.spot)
                if forwards
                else market_surface.spot
            )
            fit = self.calibrate_slice(
                sub["strike"].to_numpy(), sub["mid_iv"].to_numpy(), F, float(T), beta
            )
            fit["maturity"] = float(T)
            rows.append(fit)
        return pd.DataFrame(rows)

    def compare_with_heston(
        self, heston_surface, sabr_surface, market_surface
    ) -> pd.DataFrame:
        """Per-maturity RMSE comparison of Heston vs SABR fits (in bps)."""
        rows = []
        for i, T in enumerate(market_surface.maturities):
            rows.append(
                {
                    "maturity": T,
                    "heston_rmse_bps": np.sqrt(
                        np.nanmean(
                            (heston_surface[i] - market_surface.mid_surface()[i]) ** 2
                        )
                    )
                    * 1e4,
                    "sabr_rmse_bps": np.sqrt(
                        np.nanmean(
                            (sabr_surface[i] - market_surface.mid_surface()[i]) ** 2
                        )
                    )
                    * 1e4,
                }
            )
        return pd.DataFrame(rows)
