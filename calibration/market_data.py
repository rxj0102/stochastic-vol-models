"""Market implied-volatility surface loading, cleaning and synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["MarketIVSurface"]

_REQUIRED = ["maturity", "strike", "mid_iv"]


@dataclass
class MarketIVSurface:
    """A market implied-volatility surface in long (tidy) format.

    The underlying :attr:`data` frame has one row per quote with columns
    ``maturity, strike, bid_iv, ask_iv, mid_iv, volume, open_interest``.
    """

    data: pd.DataFrame
    spot: float

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str, spot: Optional[float] = None) -> "MarketIVSurface":
        """Load a surface from CSV (see class docstring for columns)."""
        df = pd.read_csv(path)
        cls._validate_columns(df)
        if spot is None:
            spot = float(df.attrs.get("spot", df["strike"].median()))
        return cls(df.reset_index(drop=True), spot)

    @classmethod
    def from_dict(cls, data_dict: Dict, spot: float) -> "MarketIVSurface":
        """Build a surface from a dict of equal-length columns."""
        df = pd.DataFrame(data_dict)
        cls._validate_columns(df)
        return cls(df.reset_index(drop=True), spot)

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"surface missing required columns: {missing}")

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @property
    def maturities(self) -> np.ndarray:
        """Sorted unique maturities."""
        return np.sort(self.data["maturity"].unique())

    @property
    def strikes(self) -> np.ndarray:
        """Sorted unique strikes."""
        return np.sort(self.data["strike"].unique())

    def mid_surface(self) -> np.ndarray:
        """Return the mid-IV grid ``(maturities x strikes)`` (NaN where absent)."""
        return (
            self.data.pivot_table(index="maturity", columns="strike", values="mid_iv")
            .reindex(index=self.maturities, columns=self.strikes)
            .values
        )

    def moneyness_grid(self, log: bool = True) -> np.ndarray:
        """Return strikes as moneyness (``ln(K/S0)`` if ``log`` else ``K/S0``)."""
        m = self.strikes / self.spot
        return np.log(m) if log else m

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def filter_liquid(
        self, min_volume: int = 100, min_oi: int = 500
    ) -> "MarketIVSurface":
        """Drop quotes below the given volume / open-interest thresholds."""
        df = self.data
        mask = np.ones(len(df), dtype=bool)
        if "volume" in df:
            mask &= df["volume"].fillna(0) >= min_volume
        if "open_interest" in df:
            mask &= df["open_interest"].fillna(0) >= min_oi
        return MarketIVSurface(df[mask].reset_index(drop=True), self.spot)

    def filter_no_arbitrage(self) -> "MarketIVSurface":
        """Remove calendar-spread violations in total variance.

        For each strike, total variance ``sigma^2 * T`` must be non-decreasing
        in maturity; offending rows are dropped.
        """
        df = self.data.copy()
        keep = np.ones(len(df), dtype=bool)
        for K, grp in df.groupby("strike"):
            grp = grp.sort_values("maturity")
            tv = grp["mid_iv"].values ** 2 * grp["maturity"].values
            running = -np.inf
            for pos, idx in enumerate(grp.index):
                if tv[pos] < running - 1e-10:
                    keep[df.index.get_loc(idx)] = False
                else:
                    running = max(running, tv[pos])
        return MarketIVSurface(df[keep].reset_index(drop=True), self.spot)

    def plot_surface_data(self) -> Dict[str, np.ndarray]:
        """Return ``{x: strikes, y: maturities, z: iv grid}`` for a 3-D plot."""
        return {
            "x": self.strikes,
            "y": self.maturities,
            "z": self.mid_surface(),
        }

    # ------------------------------------------------------------------
    # Synthetic surface for testing
    # ------------------------------------------------------------------
    @staticmethod
    def generate_heston_surface(
        params,
        S0: float,
        r: float,
        q: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        bid_ask_vol: float = 0.005,
    ) -> "MarketIVSurface":
        """Generate an arbitrage-free synthetic surface from Heston parameters.

        Parameters
        ----------
        params : object
            Heston parameter object (``kappa, theta, xi, rho, V0``).
        S0, r, q : float
        strikes, maturities : numpy.ndarray
        bid_ask_vol : float, optional
            Half-spread (in vols) used to populate bid/ask columns.

        Returns
        -------
        MarketIVSurface
        """
        from fourier_pricing.cos_method import COSMethodPricer
        from implied_vol.bsm_inversion import BSMInverter

        pricer = COSMethodPricer()
        inv = BSMInverter()
        rows = []
        for T in maturities:
            fwd = S0 * np.exp((r - q) * T)
            for K in strikes:
                otype = "call" if K >= fwd else "put"
                price = pricer.price(S0, float(K), float(T), r, q, params, otype)
                iv = inv.safe_implied_vol(price, S0, float(K), float(T), r, q, otype)
                if not np.isfinite(iv):
                    continue
                rows.append(
                    {
                        "maturity": float(T),
                        "strike": float(K),
                        "mid_iv": iv,
                        "bid_iv": iv - bid_ask_vol,
                        "ask_iv": iv + bid_ask_vol,
                        "volume": 1000.0,
                        "open_interest": 5000.0,
                    }
                )
        return MarketIVSurface(pd.DataFrame(rows), S0)
