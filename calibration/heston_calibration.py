"""End-to-end Heston calibration pipeline."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import pandas as pd

from fourier_pricing.cos_method import COSMethodPricer
from implied_vol.bsm_inversion import BSMInverter
from models.base import CalibrationResult
from models.heston import HestonParams

from .loss_functions import CalibrationLoss
from .market_data import MarketIVSurface
from .optimizer import DEFAULT_BOUNDS, CalibrationOptimizer

__all__ = ["HestonCalibrator"]


class HestonCalibrator:
    """Calibrate Heston parameters to a market implied-volatility surface."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.pricer = COSMethodPricer()
        self.inv = BSMInverter()
        self.loss = CalibrationLoss()
        self.optimizer = CalibrationOptimizer(seed=seed)

    # ------------------------------------------------------------------
    def _model_ivs(self, x, S0, r, q, mats, strikes):
        """Vectorised model IVs for the quote arrays given param vector ``x``."""
        params = HestonParams(*x)
        model_iv = np.empty(mats.size)
        for i in range(mats.size):
            T, K = mats[i], strikes[i]
            otype = "call" if K >= S0 * np.exp((r - q) * T) else "put"
            price = self.pricer.price(S0, K, T, r, q, params, otype)
            model_iv[i] = self.inv.safe_implied_vol(price, S0, K, T, r, q, otype)
        return model_iv

    def calibrate_joint(
        self,
        market_surface: MarketIVSurface,
        S0: float,
        r: float,
        q: float,
        loss_type: str = "bid_ask",
        bounds=None,
        de_maxiter: int = 150,
    ) -> CalibrationResult:
        """Joint calibration: one parameter set for the whole surface.

        Returns
        -------
        CalibrationResult
            The fitted :class:`HestonParams` plus fit diagnostics.
        """
        df = market_surface.data
        mats = df["maturity"].to_numpy(float)
        strikes = df["strike"].to_numpy(float)
        market_iv = df["mid_iv"].to_numpy(float)
        bid = df.get("bid_iv", market_iv - 0.005).to_numpy(float)
        ask = df.get("ask_iv", market_iv + 0.005).to_numpy(float)
        weights = df.get("volume", pd.Series(np.ones(len(df)))).to_numpy(float)

        bounds = bounds or DEFAULT_BOUNDS

        def objective(x):
            try:
                model_iv = self._model_ivs(x, S0, r, q, mats, strikes)
            except (ValueError, FloatingPointError):
                return 1e6
            return self.loss.compute(
                market_iv,
                model_iv,
                loss_type,
                weights=weights,
                bid_ivs=bid,
                ask_ivs=ask,
            )

        t0 = time.perf_counter()
        out = self.optimizer.optimize(
            objective, bounds, "two_stage", de_maxiter=de_maxiter
        )
        elapsed = time.perf_counter() - t0

        fitted = HestonParams(*out.x)
        model_iv = self._model_ivs(out.x, S0, r, q, mats, strikes)
        rmse = self.loss.rmse(market_iv, model_iv)
        return CalibrationResult(
            params=fitted,
            rmse_bps=rmse * 1e4,
            n_points=int(np.isfinite(model_iv).sum()),
            converged=out.converged,
            elapsed_seconds=elapsed,
            extra={
                "loss_type": loss_type,
                "final_loss": out.fun,
                "history": out.history,
                "feller_ratio": 2 * fitted.kappa * fitted.theta / fitted.xi**2,
            },
        )

    def calibrate_term_structure(
        self, market_surface: MarketIVSurface, S0, r, q, loss_type="rmse"
    ) -> List[CalibrationResult]:
        """Slice-by-slice calibration, one parameter set per maturity."""
        results = []
        for T in market_surface.maturities:
            sub = market_surface.data[market_surface.data["maturity"] == T]
            slice_surf = MarketIVSurface(sub.reset_index(drop=True), S0)
            results.append(
                self.calibrate_joint(slice_surf, S0, r, q, loss_type, de_maxiter=80)
            )
        return results

    def time_dependent_params(self, market_surface, S0, r, q) -> pd.DataFrame:
        """Piecewise-constant parameter term structure across maturity buckets."""
        rows = []
        for T, res in zip(
            market_surface.maturities,
            self.calibrate_term_structure(market_surface, S0, r, q),
        ):
            d = res.params.as_dict()
            d["maturity"] = T
            d["rmse_bps"] = res.rmse_bps
            rows.append(d)
        return pd.DataFrame(rows)

    def sensitivity_analysis(
        self, calibrated_params: HestonParams, market_surface, S0, r, q, bump=0.10
    ) -> pd.DataFrame:
        """Re-evaluate fit RMSE under +/- ``bump`` perturbations of each param."""
        df = market_surface.data
        mats = df["maturity"].to_numpy(float)
        strikes = df["strike"].to_numpy(float)
        market_iv = df["mid_iv"].to_numpy(float)
        base = np.array(list(calibrated_params.as_dict().values()))
        names = list(calibrated_params.as_dict().keys())
        rows = []
        for i, name in enumerate(names):
            for sign in (-1, 1):
                x = base.copy()
                x[i] *= 1.0 + sign * bump
                model_iv = self._model_ivs(x, S0, r, q, mats, strikes)
                rows.append(
                    {
                        "param": name,
                        "bump": sign * bump,
                        "rmse_bps": self.loss.rmse(market_iv, model_iv) * 1e4,
                    }
                )
        return pd.DataFrame(rows)
