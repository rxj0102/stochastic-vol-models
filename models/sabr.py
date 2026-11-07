"""SABR model (Hagan et al., 2002) — used as a comparison benchmark.

Only the implied-volatility expansion is implemented (the lognormal Hagan
formula), since SABR is used here purely to fit individual maturity slices and
contrast its smile against Heston.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ModelParams

__all__ = ["SABRParams", "hagan_lognormal_vol", "SABRModel"]


@dataclass
class SABRParams(ModelParams):
    """SABR parameters for one maturity slice.

    Parameters
    ----------
    alpha : float
        Overall volatility level (``> 0``).
    beta : float
        CEV exponent in ``[0, 1]`` (0.5 is a common equity convention).
    rho : float
        Correlation between forward and vol in ``(-1, 1)``.
    nu : float
        Vol-of-vol (``> 0``).
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(
                f"SABR beta must lie in [0, 1] (0=normal, 0.5=equity, "
                f"1=lognormal), got {self.beta}"
            )
        if self.alpha <= 0 or self.nu <= 0:
            raise ValueError("SABR alpha and nu must be strictly positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("SABR rho must lie in (-1, 1)")


def hagan_lognormal_vol(
    F: float, K: np.ndarray, T: float, alpha: float, beta: float, rho: float, nu: float
) -> np.ndarray:
    r"""Hagan (2002) lognormal implied-volatility approximation.

    Parameters
    ----------
    F : float
        Forward price.
    K : numpy.ndarray
        Strike(s).
    T : float
        Maturity.
    alpha, beta, rho, nu : float
        SABR parameters.

    Returns
    -------
    numpy.ndarray
        Black implied volatility at each strike.

    Notes
    -----
    Implements the standard expansion with the ATM limit handled separately to
    avoid the ``0/0`` in ``z/x(z)``.
    """
    K = np.asarray(K, dtype=np.float64)
    eps = 1e-12
    one_beta = 1.0 - beta

    FK = F * K
    log_FK = np.log(F / K)

    z = (nu / alpha) * (FK ** (one_beta / 2.0)) * log_FK
    # x(z)
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

    # z / x(z) with the ATM limit -> 1.
    ratio = np.where(np.abs(z) < eps, 1.0, z / np.where(np.abs(xz) < eps, eps, xz))

    FK_pow = FK ** (one_beta / 2.0)
    denom = FK_pow * (
        1.0 + (one_beta**2 / 24.0) * log_FK**2 + (one_beta**4 / 1920.0) * log_FK**4
    )

    term = (
        1.0
        + (
            (one_beta**2 / 24.0) * alpha**2 / (FK**one_beta)
            + 0.25 * rho * beta * nu * alpha / FK_pow
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        )
        * T
    )

    vol = (alpha / denom) * ratio * term

    # ATM closed form (K == F).
    atm = (alpha / F**one_beta) * (
        1.0
        + (
            (one_beta**2 / 24.0) * alpha**2 / F ** (2.0 * one_beta)
            + 0.25 * rho * beta * nu * alpha / F**one_beta
            + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
        )
        * T
    )
    vol = np.where(np.abs(log_FK) < 1e-10, atm, vol)
    return vol


class SABRModel:
    """Thin wrapper around :func:`hagan_lognormal_vol`."""

    def __init__(self, params: SABRParams) -> None:
        self.p = params

    @property
    def name(self) -> str:
        return "SABR"

    def implied_vol(self, F: float, K: np.ndarray, T: float) -> np.ndarray:
        """Black implied volatility at strikes ``K`` for forward ``F``."""
        return hagan_lognormal_vol(
            F, K, T, self.p.alpha, self.p.beta, self.p.rho, self.p.nu
        )
