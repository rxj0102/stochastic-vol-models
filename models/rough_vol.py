"""Rough volatility sketch: rBergomi-style variance (Bayer-Friz-Gatheral 2016).

A lightweight, forward-looking extension included for comparison.  The full
simulation lives in :mod:`simulation.rough_simulation`; this module only
exposes the parameter container and the deterministic forward-variance curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ModelParams

__all__ = ["RoughBergomiParams"]


@dataclass
class RoughBergomiParams(ModelParams):
    """rBergomi parameters.

    Parameters
    ----------
    xi0 : float
        Flat forward variance level (``> 0``).
    eta : float
        Volatility of (log) variance (``> 0``).
    H : float
        Hurst exponent in ``(0, 0.5)``; empirically ``H ~ 0.1``.
    rho : float
        Spot-vol correlation in ``(-1, 1)``.
    """

    xi0: float
    eta: float
    H: float
    rho: float

    def __post_init__(self) -> None:
        if not 0.0 < self.H < 0.5:
            raise ValueError("Hurst exponent H must lie in (0, 0.5)")

    def forward_variance(self, t: np.ndarray) -> np.ndarray:
        """Deterministic forward variance curve (flat at ``xi0`` here)."""
        t = np.asarray(t, dtype=np.float64)
        return np.full_like(t, self.xi0)
