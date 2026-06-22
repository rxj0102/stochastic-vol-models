"""Statistical and Black-Scholes helper functions used across the library."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "bsm_price",
    "bsm_vega",
    "forward_price",
    "moments_from_samples",
]


def forward_price(S0: float, T: float, r: float, q: float) -> float:
    """Return the forward price :math:`F = S_0 e^{(r-q)T}`."""
    return S0 * np.exp((r - q) * T)


def bsm_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    r"""Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    S0, K, T, r, q : float
        Spot, strike, maturity, rate, dividend yield.
    sigma : float
        Volatility.
    option_type : {'call', 'put'}, optional

    Returns
    -------
    float
        Option premium.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        if option_type == "call":
            return intrinsic
        return max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)


def bsm_vega(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega ``dPrice/dsigma`` (same for call and put)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    return S0 * np.exp(-q * T) * norm.pdf(d1) * sqrtT


def moments_from_samples(x: np.ndarray) -> dict:
    """Return mean, variance, skewness and excess kurtosis of a sample."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    var = x.var(ddof=1)
    sd = np.sqrt(var)
    skew = np.mean(((x - mu) / sd) ** 3)
    kurt = np.mean(((x - mu) / sd) ** 4) - 3.0
    return {
        "mean": float(mu),
        "variance": float(var),
        "skewness": float(skew),
        "kurtosis": float(kurt),
    }
