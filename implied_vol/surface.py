"""Implied volatility surface construction and interpolation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .bsm_inversion import BSMInverter

__all__ = ["IVSurface"]


class IVSurface:
    """A model or market implied-volatility surface with 2-D interpolation.

    Parameters
    ----------
    strikes : numpy.ndarray
        1-D strike grid.
    maturities : numpy.ndarray
        1-D maturity grid.
    iv : numpy.ndarray
        2-D array of implied vols ``(maturities x strikes)``.
    spot : float
        Reference spot for moneyness conversions.
    """

    def __init__(self, strikes, maturities, iv, spot: float) -> None:
        self.strikes = np.asarray(strikes, dtype=np.float64)
        self.maturities = np.asarray(maturities, dtype=np.float64)
        self.iv = np.asarray(iv, dtype=np.float64)
        self.spot = spot
        self._spline = None

    # ------------------------------------------------------------------
    @classmethod
    def from_heston(
        cls, params, S0, r, q, strikes, maturities, pricer="cos"
    ) -> "IVSurface":
        """Build a surface from Heston parameters via Fourier pricing."""
        from fourier_pricing.carr_madan import CarrMadanPricer
        from fourier_pricing.cos_method import COSMethodPricer

        engine = COSMethodPricer() if pricer == "cos" else CarrMadanPricer()
        inv = BSMInverter()
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        iv = np.empty((maturities.size, strikes.size))
        for i, T in enumerate(maturities):
            fwd = S0 * np.exp((r - q) * T)
            for j, K in enumerate(strikes):
                otype = "call" if K >= fwd else "put"
                price = engine.price(S0, float(K), float(T), r, q, params, otype)
                iv[i, j] = inv.safe_implied_vol(
                    price, S0, float(K), float(T), r, q, otype
                )
        return cls(strikes, maturities, iv, S0)

    @classmethod
    def from_market_prices(
        cls, prices, S0, strikes, maturities, r, q, option_types
    ) -> "IVSurface":
        """Build a surface from a price grid by BSM inversion."""
        inv = BSMInverter()
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        iv = np.empty_like(prices)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                ot = option_types[i][j] if np.ndim(option_types) == 2 else option_types
                iv[i, j] = inv.safe_implied_vol(
                    prices[i, j], S0, float(K), float(T), r, q, ot
                )
        return cls(strikes, maturities, iv, S0)

    # ------------------------------------------------------------------
    def _build_spline(self):
        if self._spline is None:
            iv = np.nan_to_num(self.iv, nan=float(np.nanmean(self.iv)))
            kx = min(3, self.maturities.size - 1)
            ky = min(3, self.strikes.size - 1)
            self._spline = RectBivariateSpline(
                self.maturities, self.strikes, iv, kx=max(kx, 1), ky=max(ky, 1)
            )
        return self._spline

    def iv_at(self, K: float, T: float) -> float:
        """Interpolate the implied vol at a single ``(K, T)``."""
        return float(self._build_spline()(T, K)[0, 0])

    def delta_surface(self, r: float = 0.0, q: float = 0.0) -> np.ndarray:
        """Approximate Black-Scholes call delta at each grid node."""
        from scipy.stats import norm

        out = np.empty_like(self.iv)
        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                sig = self.iv[i, j]
                if not np.isfinite(sig) or sig <= 0:
                    out[i, j] = np.nan
                    continue
                d1 = (np.log(self.spot / K) + (r - q + 0.5 * sig**2) * T) / (
                    sig * np.sqrt(T)
                )
                out[i, j] = np.exp(-q * T) * norm.cdf(d1)
        return out

    def plot_3d_data(self) -> Dict[str, np.ndarray]:
        """Return ``{x, y, z}`` for a Plotly ``go.Surface``."""
        return {"x": self.strikes, "y": self.maturities, "z": self.iv}

    def plot_slice(self, T: float) -> Dict[str, np.ndarray]:
        """Return the smile (strikes, ivs) closest to maturity ``T``."""
        i = int(np.argmin(np.abs(self.maturities - T)))
        return {
            "strikes": self.strikes,
            "iv": self.iv[i],
            "maturity": float(self.maturities[i]),
        }
