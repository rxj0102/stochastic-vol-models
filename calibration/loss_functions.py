"""Calibration objective (loss) functions on implied volatilities or prices."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["CalibrationLoss"]


class CalibrationLoss:
    """A family of calibration loss functions.

    Each method consumes flattened arrays of market and model implied vols
    (NaNs are ignored) and returns a scalar to be minimised.
    """

    @staticmethod
    def _clean(market, model):
        market = np.asarray(market, dtype=np.float64).ravel()
        model = np.asarray(model, dtype=np.float64).ravel()
        mask = np.isfinite(market) & np.isfinite(model)
        return market[mask], model[mask], mask

    def rmse(self, market_ivs, model_ivs) -> float:
        """Root-mean-square error in implied-vol points."""
        m, mo, _ = self._clean(market_ivs, model_ivs)
        if m.size == 0:
            return np.inf
        return float(np.sqrt(np.mean((mo - m) ** 2)))

    def vwrmse(self, market_ivs, model_ivs, weights) -> float:
        """Volume-weighted RMSE."""
        m, mo, mask = self._clean(market_ivs, model_ivs)
        w = np.asarray(weights, dtype=np.float64).ravel()[mask]
        if m.size == 0 or w.sum() == 0:
            return np.inf
        w = w / w.sum()
        return float(np.sqrt(np.sum(w * (mo - m) ** 2)))

    def bid_ask_normalized(self, market_ivs, model_ivs, bid_ivs, ask_ivs) -> float:
        """Mean absolute error normalised by the bid-ask spread.

        Model errors inside the spread contribute little; errors outside it are
        penalised proportionally to how far outside they fall.
        """
        m, mo, mask = self._clean(market_ivs, model_ivs)
        bid = np.asarray(bid_ivs, dtype=np.float64).ravel()[mask]
        ask = np.asarray(ask_ivs, dtype=np.float64).ravel()[mask]
        spread = np.maximum(ask - bid, 1e-4)
        if m.size == 0:
            return np.inf
        return float(np.mean(np.abs(mo - m) / spread))

    def relative_rmse(self, market_ivs, model_ivs) -> float:
        """RMSE of relative errors ``(model - market)/market``."""
        m, mo, _ = self._clean(market_ivs, model_ivs)
        if m.size == 0:
            return np.inf
        return float(np.sqrt(np.mean(((mo - m) / m) ** 2)))

    def price_rmse(self, market_prices, model_prices) -> float:
        """RMSE on prices (avoids BSM-inversion noise for deep wings)."""
        m, mo, _ = self._clean(market_prices, model_prices)
        if m.size == 0:
            return np.inf
        return float(np.sqrt(np.mean((mo - m) ** 2)))

    def compute(
        self,
        market_ivs,
        model_ivs,
        loss_type: str = "rmse",
        weights: Optional[np.ndarray] = None,
        bid_ivs: Optional[np.ndarray] = None,
        ask_ivs: Optional[np.ndarray] = None,
    ) -> float:
        """Dispatch to the named loss function."""
        if loss_type == "rmse":
            return self.rmse(market_ivs, model_ivs)
        if loss_type in ("vwrmse", "volume_weighted"):
            return self.vwrmse(market_ivs, model_ivs, weights)
        if loss_type in ("bid_ask", "bid_ask_normalized"):
            return self.bid_ask_normalized(market_ivs, model_ivs, bid_ivs, ask_ivs)
        if loss_type in ("relative", "relative_rmse"):
            return self.relative_rmse(market_ivs, model_ivs)
        if loss_type in ("price", "price_rmse"):
            return self.price_rmse(market_ivs, model_ivs)
        raise ValueError(f"unknown loss_type {loss_type!r}")

    def weighted_by_maturity(self, market_ivs, model_ivs, maturity_weights) -> float:
        """RMSE with per-row maturity weights."""
        return self.vwrmse(market_ivs, model_ivs, maturity_weights)
