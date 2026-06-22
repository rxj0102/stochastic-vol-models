"""Carr-Madan FFT option pricing (Carr & Madan, 1999).

The Carr-Madan method prices an entire strike smile for one maturity with a
single Fast Fourier Transform.  The call payoff is damped by
:math:`e^{\\alpha k}` (``k = ln K``) to make the Fourier transform integrable.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from characteristic_functions.heston_cf import HestonCharacteristicFunction

__all__ = ["CarrMadanPricer", "DEFAULT_N_FFT", "DEFAULT_ALPHA_CM"]

DEFAULT_N_FFT = 4096
DEFAULT_ALPHA_CM = 1.5


class CarrMadanPricer:
    """Carr-Madan FFT pricer for Heston European options.

    Parameters
    ----------
    formulation : str, optional
        Forwarded to :class:`HestonCharacteristicFunction`.
    """

    def __init__(self, formulation: str = "little_trap") -> None:
        self.cf = HestonCharacteristicFunction(formulation)

    def price_fft(
        self,
        S0,
        T,
        r,
        q,
        params,
        alpha: float = DEFAULT_ALPHA_CM,
        N: int = DEFAULT_N_FFT,
        eta: float = 0.25,
    ):
        r"""Compute the full call smile for one maturity via FFT.

        Parameters
        ----------
        S0, T, r, q : float
            Contract data (no strike — the whole smile is returned).
        params : object
            Object with ``kappa, theta, xi, rho, V0`` attributes.
        alpha : float, optional
            Damping parameter (positive for calls).
        N : int, optional
            FFT size (power of two).
        eta : float, optional
            Frequency-domain grid spacing.

        Returns
        -------
        tuple of numpy.ndarray
            ``(strikes, call_prices)`` sampled on the FFT log-strike grid.

        Notes
        -----
        With :math:`\lambda = 2\pi/(N\eta)` the log-strike grid is
        :math:`k_j = -N\lambda/2 + j\lambda`.  Simpson weights are applied to
        the integrand for higher-order accuracy.
        """
        kappa, theta, xi, rho, V0 = (
            params.kappa,
            params.theta,
            params.xi,
            params.rho,
            params.V0,
        )
        lam = 2.0 * np.pi / (N * eta)
        b = N * lam / 2.0

        j = np.arange(N)
        v = j * eta  # frequency grid

        # psi(v) = e^{-rT} phi(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha+1)v)
        u = v - (alpha + 1.0) * 1j
        phi = self.cf.cf(u, S0, V0, T, r, q, kappa, theta, xi, rho)
        denom = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
        psi = np.exp(-r * T) * phi / denom

        # Simpson's rule weights.
        simpson = (3.0 + (-1.0) ** (j + 1)) / 3.0
        simpson[0] = 1.0 / 3.0

        integrand = np.exp(1j * b * v) * psi * eta * simpson
        fft_vals = np.fft.fft(integrand).real

        k = -b + lam * j
        call_prices = np.exp(-alpha * k) / np.pi * fft_vals
        # The raw CF already carries ln(S0); k is the actual log-strike.
        strikes = np.exp(k)
        return strikes, call_prices

    def _smile_spline(self, S0, T, r, q, params, alpha, N, eta):
        strikes, calls = self.price_fft(S0, T, r, q, params, alpha, N, eta)
        # Keep a sensible, monotone region around the spot for interpolation.
        mask = (strikes > 0.1 * S0) & (strikes < 10.0 * S0) & np.isfinite(calls)
        ks, cs = strikes[mask], calls[mask]
        order = np.argsort(ks)
        return CubicSpline(ks[order], cs[order])

    def price(
        self,
        S0,
        K,
        T,
        r,
        q,
        params,
        option_type: str = "call",
        alpha: float = DEFAULT_ALPHA_CM,
        N: int = DEFAULT_N_FFT,
        eta: float = 0.25,
    ) -> float:
        """Price a single option, interpolating the FFT smile at ``K``.

        Puts are obtained from the call price by put-call parity.
        """
        spline = self._smile_spline(S0, T, r, q, params, alpha, N, eta)
        call = float(spline(K))
        if option_type == "call":
            return max(call, 0.0)
        # put-call parity: P = C - S0 e^{-qT} + K e^{-rT}
        put = call - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        return max(put, 0.0)

    def price_surface(
        self,
        S0,
        strikes,
        maturities,
        r,
        q,
        params,
        option_type="call",
        alpha=DEFAULT_ALPHA_CM,
    ) -> np.ndarray:
        """Price a grid of options (one FFT per maturity)."""
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        out = np.empty((maturities.size, strikes.size))
        for i, T in enumerate(maturities):
            spline = self._smile_spline(S0, T, r, q, params, alpha, DEFAULT_N_FFT, 0.25)
            calls = spline(strikes)
            if option_type == "call":
                out[i] = np.maximum(calls, 0.0)
            else:
                out[i] = np.maximum(
                    calls - S0 * np.exp(-q * T) + strikes * np.exp(-r * T), 0.0
                )
        return out
