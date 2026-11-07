"""Black-Scholes implied volatility inversion.

A robust Newton-Raphson solver with a Brenner-Subrahmanyam starting guess and a
bisection fallback, plus a ``safe_implied_vol`` wrapper that returns ``NaN``
instead of raising for arbitrage-violating or degenerate inputs.
"""

from __future__ import annotations

import numpy as np

from utils.statistics import bsm_price, bsm_vega

__all__ = ["BSMInverter"]

_MAX_ITER = 100
_TOL = 1e-8


class BSMInverter:
    """Extract Black-Scholes implied volatility from option prices."""

    @staticmethod
    def _initial_guess(price, S0, K, T, r, q) -> float:
        """Brenner-Subrahmanyam ATM approximation, clamped to a sane range."""
        F = S0 * np.exp((r - q) * T)
        # Brenner-Subrahmanyam: sigma ~ sqrt(2pi/T) * price / S0.
        guess = np.sqrt(2.0 * np.pi / T) * price / S0 if T > 0 else 0.2
        return float(min(max(guess, 1e-3), 5.0))

    def implied_vol(
        self, price, S0, K, T, r, q, option_type="call", method="newton"
    ) -> float:
        r"""Return the implied volatility solving ``BSM(sigma) = price``.

        Parameters
        ----------
        price : float
            Observed option premium.
        S0, K, T, r, q : float
        option_type : {'call', 'put'}, optional
        method : {'newton', 'bisection'}, optional

        Returns
        -------
        float
            Implied volatility.

        Raises
        ------
        ValueError
            If the price violates static arbitrage bounds (use
            :meth:`safe_implied_vol` to get ``NaN`` instead).
        """
        # No-arbitrage bounds.
        if option_type == "call":
            lower = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            upper = S0 * np.exp(-q * T)
        else:
            lower = max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
            upper = K * np.exp(-r * T)
        tol = 1e-7 * max(S0, K) + 1e-10
        if price < lower - tol or price > upper + tol:
            raise ValueError(
                f"price {price:.6g} outside no-arbitrage bounds "
                f"[{lower:.6g}, {upper:.6g}]"
            )
        # Clamp tiny rounding excursions back into the admissible band.
        price = min(max(price, lower + 1e-12), upper - 1e-12)

        if method == "bisection":
            return self._bisection(price, S0, K, T, r, q, option_type)

        sigma = self._initial_guess(max(price - lower, 1e-8), S0, K, T, r, q)
        for _ in range(_MAX_ITER):
            model = bsm_price(S0, K, T, r, q, sigma, option_type)
            diff = model - price
            if abs(diff) < _TOL:
                return float(sigma)
            vega = bsm_vega(S0, K, T, r, q, sigma)
            if vega < 1e-10:
                break  # fall back to bisection
            sigma -= diff / vega
            if sigma <= 0 or sigma > 10:
                break
        return self._bisection(price, S0, K, T, r, q, option_type)

    @staticmethod
    def _bisection(price, S0, K, T, r, q, option_type, lo=1e-6, hi=10.0) -> float:
        """Bracketed bisection fallback (always converges if a root exists)."""
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            val = bsm_price(S0, K, T, r, q, mid, option_type) - price
            if abs(val) < _TOL:
                return float(mid)
            if val > 0:
                hi = mid
            else:
                lo = mid
        return float(0.5 * (lo + hi))

    def implied_vol_batch(
        self, prices, S0, strikes, T, r, q, option_types
    ) -> np.ndarray:
        """Vectorised inversion over arrays of prices/strikes."""
        prices = np.asarray(prices, dtype=np.float64)
        strikes = np.asarray(strikes, dtype=np.float64)
        if isinstance(option_types, str):
            option_types = [option_types] * prices.size
        out = np.empty(prices.size)
        for i in range(prices.size):
            out[i] = self.safe_implied_vol(
                float(prices[i]), S0, float(strikes[i]), T, r, q, option_types[i]
            )
        return out

    def safe_implied_vol(self, price, S0, K, T, r, q, option_type="call") -> float:
        """Like :meth:`implied_vol` but returns ``NaN`` instead of raising."""
        try:
            return self.implied_vol(price, S0, K, T, r, q, option_type)
        except (ValueError, FloatingPointError):
            return float("nan")
