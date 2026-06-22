"""Dupire local volatility surface (Dupire, 1994).

Given a (model or market) implied-volatility surface, the Dupire equation
recovers the local volatility ``sigma_loc(K, T)`` consistent with it.  This is
used in the model-comparison module to contrast Heston's forward smile dynamics
with the (static-smile) local-vol picture.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DupireLocalVol"]


class DupireLocalVol:
    r"""Local volatility from a total-variance surface via the Dupire formula.

    The implementation works in terms of total implied variance
    :math:`w(k, T) = \sigma_{\mathrm{imp}}(k, T)^2 T` as a function of
    log-moneyness :math:`k = \ln(K/F)`, which is the numerically stable
    parameterisation (Gatheral, 2006):

    .. math::

        \sigma_{\mathrm{loc}}^2 =
        \frac{\partial_T w}
        {1 - \frac{k}{w}\partial_k w
        + \frac14\left(-\frac14 - \frac1w + \frac{k^2}{w^2}\right)(\partial_k w)^2
        + \frac12\partial_{kk} w}.
    """

    def __init__(
        self,
        log_moneyness: np.ndarray,
        maturities: np.ndarray,
        total_variance: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        log_moneyness : numpy.ndarray
            1-D grid of ``k = ln(K/F)``.
        maturities : numpy.ndarray
            1-D grid of maturities (increasing).
        total_variance : numpy.ndarray
            2-D array ``w[i, j]`` = total variance at ``(maturities[i], k[j])``.
        """
        self.k = np.asarray(log_moneyness, dtype=np.float64)
        self.T = np.asarray(maturities, dtype=np.float64)
        self.w = np.asarray(total_variance, dtype=np.float64)

    def local_vol_surface(self) -> np.ndarray:
        """Return the local-volatility surface on the input grid.

        Returns
        -------
        numpy.ndarray
            2-D array of local volatilities, same shape as ``total_variance``.
            Cells where the denominator is non-positive (arbitrage / boundary)
            are returned as ``NaN``.
        """
        k, T, w = self.k, self.T, self.w
        dwdT = np.gradient(w, T, axis=0)
        dwdk = np.gradient(w, k, axis=1)
        d2wdk2 = np.gradient(dwdk, k, axis=1)

        kk = k[None, :]
        w_safe = np.where(w > 1e-8, w, np.nan)

        denom = (
            1.0
            - (kk / w_safe) * dwdk
            + 0.25 * (-0.25 - 1.0 / w_safe + kk**2 / w_safe**2) * dwdk**2
            + 0.5 * d2wdk2
        )
        loc_var = dwdT / denom
        loc_var = np.where((denom > 0) & (loc_var > 0), loc_var, np.nan)
        return np.sqrt(loc_var)
