"""Calibration quality diagnostics: residuals, stability and fit reports."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from models.base import CalibrationResult

__all__ = ["CalibrationDiagnostics"]


class CalibrationDiagnostics:
    """Post-calibration analysis helpers."""

    @staticmethod
    def residual_surface(market_ivs: np.ndarray, model_ivs: np.ndarray) -> np.ndarray:
        """Return ``(model - market)`` in vol points (same shape as inputs)."""
        return np.asarray(model_ivs) - np.asarray(market_ivs)

    @staticmethod
    def smile_overlay(market_ivs, model_ivs, maturities, strikes) -> Dict:
        """Package market/model smiles per maturity for plotting."""
        return {
            "maturities": np.asarray(maturities),
            "strikes": np.asarray(strikes),
            "market": np.asarray(market_ivs),
            "model": np.asarray(model_ivs),
        }

    @staticmethod
    def error_heatmap_data(residuals: np.ndarray) -> np.ndarray:
        """Return residuals in basis points for a heatmap."""
        return np.asarray(residuals) * 1e4

    @staticmethod
    def parameter_stability(
        calibration_history: List[CalibrationResult],
    ) -> pd.DataFrame:
        """Tabulate fitted parameters across a sequence of calibrations."""
        rows = []
        for i, res in enumerate(calibration_history):
            d = dict(res.params.as_dict())
            d["calibration"] = i
            d["rmse_bps"] = res.rmse_bps
            rows.append(d)
        return pd.DataFrame(rows)

    @staticmethod
    def fit_quality_report(result: CalibrationResult) -> Dict[str, float]:
        """Summarise a calibration result as a flat dict."""
        p = result.params.as_dict()
        feller = 2 * p["kappa"] * p["theta"] / p["xi"] ** 2
        return {
            "rmse_bps": result.rmse_bps,
            "n_points": result.n_points,
            "converged": result.converged,
            "feller_ratio": feller,
            "feller_satisfied": feller >= 1.0,
            "elapsed_seconds": result.elapsed_seconds,
            **{f"param_{k}": v for k, v in p.items()},
        }

    @staticmethod
    def overfitting_check(
        train_rmse_bps: float, test_rmse_bps: float
    ) -> Dict[str, float]:
        """Compare in-sample and out-of-sample RMSE."""
        return {
            "train_rmse_bps": train_rmse_bps,
            "test_rmse_bps": test_rmse_bps,
            "ratio": test_rmse_bps / train_rmse_bps if train_rmse_bps else np.inf,
        }
