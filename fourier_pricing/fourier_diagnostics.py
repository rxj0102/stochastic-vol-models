"""Cross-method diagnostics for the Fourier pricers."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import pandas as pd

from characteristic_functions.heston_cf import HestonCharacteristicFunction

from .carr_madan import CarrMadanPricer
from .cos_method import COSMethodPricer
from .lewis_method import LewisMethodPricer

__all__ = ["FourierDiagnostics"]


class FourierDiagnostics:
    """Compare convergence, speed and accuracy of the Fourier pricers."""

    def __init__(self, formulation: str = "little_trap") -> None:
        self.cm = CarrMadanPricer(formulation)
        self.cos = COSMethodPricer(formulation)
        self.lewis = LewisMethodPricer(formulation)
        self.cf = HestonCharacteristicFunction(formulation)

    def method_comparison(
        self, S0, K, T, r, q, params, option_type="call"
    ) -> pd.DataFrame:
        """Return a DataFrame of price and elapsed time for each method."""
        rows = []
        for name, fn in (
            (
                "Carr-Madan FFT",
                lambda: self.cm.price(S0, K, T, r, q, params, option_type),
            ),
            ("COS", lambda: self.cos.price(S0, K, T, r, q, params, option_type)),
            ("Lewis", lambda: self.lewis.price(S0, K, T, r, q, params, option_type)),
        ):
            t0 = time.perf_counter()
            price = fn()
            dt = time.perf_counter() - t0
            rows.append({"method": name, "price": price, "time_ms": dt * 1e3})
        return pd.DataFrame(rows)

    def cos_convergence_plot(
        self, S0, K, T, r, q, params, option_type="call"
    ) -> Dict[str, np.ndarray]:
        """Data for COS price vs ``N`` (exponential convergence)."""
        return self.cos.convergence_test(S0, K, T, r, q, params, option_type)

    def fft_grid_sensitivity(
        self, S0, K, T, r, q, params, eta_range=None
    ) -> Dict[str, np.ndarray]:
        """Carr-Madan price vs the frequency grid spacing ``eta``."""
        if eta_range is None:
            eta_range = np.linspace(0.1, 0.5, 9)
        prices = np.array(
            [
                self.cm.price(S0, K, T, r, q, params, "call", eta=float(e))
                for e in eta_range
            ]
        )
        return {"eta": np.asarray(eta_range), "price": prices}

    def truncation_range_test(
        self, S0, K, T, r, q, params, L_range=None
    ) -> Dict[str, np.ndarray]:
        """COS price vs the cumulant truncation multiplier ``L``."""
        if L_range is None:
            L_range = [6, 8, 10, 12, 14, 16]
        prices = np.array(
            [
                self.cos.price(S0, K, T, r, q, params, "call", L=float(L))
                for L in L_range
            ]
        )
        return {"L": np.asarray(L_range), "price": prices}

    def integration_contour_plot(
        self, S0, V0, T, r, q, params, u_range=None
    ) -> Dict[str, np.ndarray]:
        """Magnitude of the CF along the Lewis contour ``Im(u) = -1/2``."""
        if u_range is None:
            u_range = np.linspace(-50, 50, 1001)
        u_range = np.asarray(u_range)
        phi = self.cf.cf(
            u_range - 0.5j,
            S0,
            params.V0,
            T,
            r,
            q,
            params.kappa,
            params.theta,
            params.xi,
            params.rho,
        )
        return {"u": u_range, "abs": np.abs(phi), "real": phi.real, "imag": phi.imag}
