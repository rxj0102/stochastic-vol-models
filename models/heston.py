"""Heston (1993) stochastic volatility model.

The model couples a geometric-Brownian spot to a Cox-Ingersoll-Ross (CIR)
variance process:

.. math::

    dS_t &= (r-q) S_t\\,dt + \\sqrt{V_t}\\,S_t\\,dW^S_t \\\\
    dV_t &= \\kappa(\\theta - V_t)\\,dt + \\xi\\sqrt{V_t}\\,dW^V_t \\\\
    d\\langle W^S, W^V\\rangle_t &= \\rho\\,dt.

This class is the user-facing façade: it owns a :class:`HestonParams` instance
and delegates pricing to the COS method, simulation to the QE scheme, and
characteristic-function evaluation to
:class:`~characteristic_functions.heston_cf.HestonCharacteristicFunction`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from characteristic_functions.heston_cf import HestonCharacteristicFunction

from .base import ModelParams, StochVolModel

__all__ = ["HestonParams", "HestonModel", "FELLER_THRESHOLD"]

FELLER_THRESHOLD = 2.0


@dataclass
class HestonParams(ModelParams):
    """Container for the five Heston parameters.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of variance (``> 0``).
    theta : float
        Long-run variance (``> 0``); note ``theta = sigma_LR**2``.
    xi : float
        Volatility of variance / vol-of-vol (``> 0``).
    rho : float
        Spot-vol correlation in ``(-1, 1)``.
    V0 : float
        Initial variance (``> 0``); note ``V0 = sigma0**2``.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    V0: float

    def __post_init__(self) -> None:
        if min(self.kappa, self.theta, self.xi, self.V0) <= 0:
            raise ValueError("kappa, theta, xi, V0 must all be strictly positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")


class HestonModel(StochVolModel):
    """High-level interface to the Heston model.

    Parameters
    ----------
    params : HestonParams
        The model parameters.
    formulation : str, optional
        Characteristic-function formulation, ``'little_trap'`` (default) or
        ``'original'``.
    validate : bool, optional
        If ``True`` (default), emit warnings for parameter regions that are
        known to be numerically or economically delicate.
    """

    def __init__(
        self,
        params: HestonParams,
        formulation: str = "little_trap",
        validate: bool = True,
    ) -> None:
        self._p = params
        self.formulation = formulation
        self._cf = HestonCharacteristicFunction(formulation)
        if validate:
            self._validate()

    # ------------------------------------------------------------------
    # Validation and derived quantities
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if self.feller_ratio < 1.0:
            warnings.warn(
                f"Feller condition violated (2*kappa*theta/xi^2 = "
                f"{self.feller_ratio:.3f} < 1): variance may hit zero.",
                RuntimeWarning,
                stacklevel=2,
            )
        if abs(self._p.rho) > 0.95:
            warnings.warn(
                "|rho| > 0.95: near-perfect correlation may cause numerical " "issues.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._p.xi > 2.0:
            warnings.warn(
                "xi > 2: extreme vol-of-vol, calibration likely unstable.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def feller_ratio(self) -> float:
        """Return ``2*kappa*theta / xi**2`` (``>= 1`` means Feller satisfied)."""
        return 2.0 * self._p.kappa * self._p.theta / self._p.xi**2

    @property
    def feller_satisfied(self) -> bool:
        """Whether the Feller condition ``2*kappa*theta >= xi**2`` holds."""
        return self.feller_ratio >= 1.0

    @property
    def long_run_vol(self) -> float:
        """Long-run volatility ``sqrt(theta)``."""
        return float(np.sqrt(self._p.theta))

    @property
    def initial_vol(self) -> float:
        """Initial volatility ``sqrt(V0)``."""
        return float(np.sqrt(self._p.V0))

    @property
    def half_life(self) -> float:
        """Variance mean-reversion half-life ``ln(2)/kappa`` (years)."""
        return float(np.log(2.0) / self._p.kappa)

    @property
    def vol_of_vol(self) -> float:
        """Normalised vol-of-vol ``xi / sqrt(2*kappa*theta)``."""
        return float(self._p.xi / np.sqrt(2.0 * self._p.kappa * self._p.theta))

    # ------------------------------------------------------------------
    # StochVolModel interface
    # ------------------------------------------------------------------
    @property
    def params(self) -> Dict[str, float]:
        return self._p.as_dict()

    @property
    def heston_params(self) -> HestonParams:
        """The underlying :class:`HestonParams` dataclass."""
        return self._p

    @property
    def name(self) -> str:
        return "Heston"

    def characteristic_function(
        self, u: np.ndarray, S0: float, K: float, T: float, r: float, q: float
    ) -> np.ndarray:
        """Evaluate the Heston log-price characteristic function at ``u``."""
        p = self._p
        return self._cf.cf(u, S0, p.V0, T, r, q, p.kappa, p.theta, p.xi, p.rho)

    def price_vanilla(self, S0, K, T, r, q, option_type: str = "call") -> float:
        """Price a European vanilla option via the COS method."""
        from fourier_pricing.cos_method import COSMethodPricer

        pricer = COSMethodPricer(self.formulation)
        return pricer.price(S0, K, T, r, q, self._p, option_type)

    def implied_vol_surface(self, S0, strikes, maturities, r, q) -> np.ndarray:
        """Return a 2-D implied-vol surface ``(maturities x strikes)``."""
        from fourier_pricing.cos_method import COSMethodPricer
        from implied_vol.bsm_inversion import BSMInverter

        pricer = COSMethodPricer(self.formulation)
        inv = BSMInverter()
        strikes = np.asarray(strikes, dtype=np.float64)
        maturities = np.asarray(maturities, dtype=np.float64)
        out = np.empty((maturities.size, strikes.size))
        F0 = S0
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Use OTM option for a numerically well-conditioned inversion.
                otype = "call" if K >= F0 * np.exp((r - q) * T) else "put"
                price = pricer.price(S0, K, T, r, q, self._p, otype)
                out[i, j] = inv.safe_implied_vol(price, S0, K, T, r, q, otype)
        return out

    def simulate(
        self, S0, V0, T, n_paths, n_steps, seed=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate spot/variance paths with the QE scheme."""
        from simulation.heston_qe import HestonQESimulator

        sim = HestonQESimulator()
        return sim.simulate(S0, V0, T, 0.0, 0.0, self._p, n_paths, n_steps, seed)
