"""Static arbitrage checks on implied-volatility surfaces.

Detects calendar-spread, butterfly (convexity) and call-spread violations, and
extracts the Breeden-Litzenberger risk-neutral density.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from utils.statistics import bsm_price

__all__ = ["ArbitrageFreeChecker"]


class ArbitrageFreeChecker:
    """Detect static-arbitrage violations on an IV surface."""

    def __init__(self, spot: float, r: float = 0.0, q: float = 0.0) -> None:
        self.spot = spot
        self.r = r
        self.q = q

    # ------------------------------------------------------------------
    def check_calendar_spread(
        self, strikes: np.ndarray, maturities: np.ndarray, iv: np.ndarray
    ) -> np.ndarray:
        """Boolean mask of calendar-spread violations.

        Total variance ``w = sigma^2 * T`` must be non-decreasing in maturity
        at each strike; ``True`` flags a decrease.
        """
        maturities = np.asarray(maturities, dtype=float)
        w = iv**2 * maturities[:, None]
        viol = np.zeros_like(iv, dtype=bool)
        viol[1:, :] = np.diff(w, axis=0) < -1e-10
        return viol

    def check_butterfly(
        self, iv_slice: np.ndarray, strikes: np.ndarray, T: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Butterfly check and risk-neutral density for one maturity slice.

        Returns
        -------
        tuple of numpy.ndarray
            ``(violations_mask, density)`` where ``density`` is the
            Breeden-Litzenberger risk-neutral PDF
            :math:`e^{rT}\partial^2 C/\partial K^2` on the interior strikes.
        """
        strikes = np.asarray(strikes, dtype=float)
        calls = np.array(
            [
                bsm_price(self.spot, K, T, self.r, self.q, sig, "call")
                for K, sig in zip(strikes, iv_slice)
            ]
        )
        d2 = np.full(strikes.size, np.nan)
        d2[1:-1] = (calls[2:] - 2.0 * calls[1:-1] + calls[:-2]) / np.diff(strikes)[
            :-1
        ] ** 2
        density = np.exp(self.r * T) * d2
        viol = density < -1e-8
        return viol, density

    def check_call_spread(
        self, iv_slice: np.ndarray, strikes: np.ndarray, T: float
    ) -> np.ndarray:
        """Mask where call prices fail to be (weakly) decreasing in strike."""
        strikes = np.asarray(strikes, dtype=float)
        calls = np.array(
            [
                bsm_price(self.spot, K, T, self.r, self.q, sig, "call")
                for K, sig in zip(strikes, iv_slice)
            ]
        )
        viol = np.zeros(strikes.size, dtype=bool)
        viol[1:] = np.diff(calls) > 1e-8
        return viol

    def full_arbitrage_report(
        self, strikes: np.ndarray, maturities: np.ndarray, iv: np.ndarray
    ) -> Dict[str, object]:
        """Aggregate counts of each violation type across the surface."""
        cal = self.check_calendar_spread(strikes, maturities, iv)
        bf_count = 0
        cs_count = 0
        neg_density = False
        for i, T in enumerate(maturities):
            vb, dens = self.check_butterfly(iv[i], strikes, float(T))
            bf_count += int(np.nansum(vb))
            cs_count += int(self.check_call_spread(iv[i], strikes, float(T)).sum())
            if np.any(dens[np.isfinite(dens)] < -1e-8):
                neg_density = True
        return {
            "calendar_violations": int(cal.sum()),
            "butterfly_violations": bf_count,
            "call_spread_violations": cs_count,
            "negative_density": neg_density,
            "arbitrage_free": bool(cal.sum() == 0 and bf_count == 0 and cs_count == 0),
        }
