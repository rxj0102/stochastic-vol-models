"""Compare Heston against SABR and flat Black-Scholes on a market surface."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd

from calibration.sabr_calibration import SABRCalibrator
from implied_vol.surface import IVSurface
from models.sabr import hagan_lognormal_vol

__all__ = ["ModelComparison"]


class ModelComparison:
    """Side-by-side comparison of Heston, SABR and constant-vol models."""

    def comparison_table(self, market_surface, heston_params, S0, r, q) -> pd.DataFrame:
        """Per-model RMSE (bps), parameter count and pricing-speed summary."""
        mid = market_surface.mid_surface()
        strikes = market_surface.strikes
        mats = market_surface.maturities

        # Heston.
        t0 = time.perf_counter()
        h_surf = IVSurface.from_heston(heston_params, S0, r, q, strikes, mats).iv
        h_time = time.perf_counter() - t0
        h_rmse = np.sqrt(np.nanmean((h_surf - mid) ** 2)) * 1e4

        # SABR slice-by-slice.
        sabr = SABRCalibrator()
        t0 = time.perf_counter()
        sabr_params = sabr.calibrate_surface(market_surface)
        s_surf = np.full_like(mid, np.nan)
        for i, T in enumerate(mats):
            row = sabr_params.iloc[i]
            F = S0 * np.exp((r - q) * T)
            s_surf[i] = hagan_lognormal_vol(
                F, strikes, T, row.alpha, row.beta, row.rho, row.nu
            )
        s_time = time.perf_counter() - t0
        s_rmse = np.sqrt(np.nanmean((s_surf - mid) ** 2)) * 1e4

        # Flat BSM (single ATM vol).
        atm = np.nanmean(mid)
        b_rmse = np.sqrt(np.nanmean((atm - mid) ** 2)) * 1e4

        return pd.DataFrame(
            {
                "model": ["Heston", "SABR (per-slice)", "Flat BSM"],
                "rmse_bps": [h_rmse, s_rmse, b_rmse],
                "n_params": [5, 3 * len(mats), 1],
                "fit_time_s": [h_time, s_time, 0.0],
            }
        )

    def smile_overlay_all_models(
        self, market_surface, slice_T, heston_params, S0, r, q
    ) -> Dict[str, np.ndarray]:
        """Market / Heston / SABR smiles at a single maturity for plotting."""
        strikes = market_surface.strikes
        i = int(np.argmin(np.abs(market_surface.maturities - slice_T)))
        T = market_surface.maturities[i]
        market = market_surface.mid_surface()[i]

        heston = IVSurface.from_heston(
            heston_params, S0, r, q, strikes, np.array([T])
        ).iv[0]

        sabr = SABRCalibrator()
        sub = market_surface.data[market_surface.data["maturity"] == T]
        F = S0 * np.exp((r - q) * T)
        fit = sabr.calibrate_slice(
            sub["strike"].to_numpy(), sub["mid_iv"].to_numpy(), F, float(T)
        )
        sabr_iv = hagan_lognormal_vol(
            F, strikes, T, fit["alpha"], fit["beta"], fit["rho"], fit["nu"]
        )

        return {
            "strikes": strikes,
            "maturity": float(T),
            "market": market,
            "heston": heston,
            "sabr": sabr_iv,
        }
