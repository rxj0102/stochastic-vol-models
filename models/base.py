"""Abstract base classes and shared data containers for stochastic volatility models.

This module defines the common interface implemented by every model in the
library (Heston, SABR, local volatility, rough volatility) together with the
lightweight dataclasses used to pass parameters and results around.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "ModelParams",
    "PriceResult",
    "CalibrationResult",
    "StochVolModel",
]


@dataclass
class ModelParams:
    """Base container for model parameters.

    Subclasses (e.g. :class:`~models.heston.HestonParams`) add the concrete
    fields.  The base class only provides a generic ``as_dict`` helper used by
    the calibration and plotting layers.
    """

    def as_dict(self) -> Dict[str, float]:
        """Return the parameters as an ordered ``dict``.

        Returns
        -------
        dict
            Mapping of parameter name to value.
        """
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class PriceResult:
    """Result of a single vanilla option price computation.

    Parameters
    ----------
    price : float
        Option premium (undiscounted units of the spot).
    implied_vol : float
        Black-Scholes implied volatility corresponding to ``price``.
    method : str
        Name of the pricing method used (e.g. ``"cos"``, ``"carr_madan"``).
    elapsed_seconds : float
        Wall-clock time taken to compute the price.
    """

    price: float
    implied_vol: float
    method: str
    elapsed_seconds: float


@dataclass
class CalibrationResult:
    """Outcome of a model calibration to a market implied-volatility surface.

    Parameters
    ----------
    params : ModelParams
        The fitted parameter set.
    rmse_bps : float
        Root-mean-square error of the fit in basis points of implied vol.
    n_points : int
        Number of market quotes used in the objective.
    converged : bool
        Whether the optimiser reported convergence.
    elapsed_seconds : float
        Total wall-clock time of the calibration.
    extra : dict, optional
        Free-form diagnostics (per-stage losses, iteration counts, ...).
    """

    params: ModelParams
    rmse_bps: float
    n_points: int
    converged: bool
    elapsed_seconds: float
    extra: Dict[str, object] = field(default_factory=dict)


class StochVolModel(abc.ABC):
    """Abstract base class for stochastic volatility models.

    Concrete subclasses must implement the characteristic function, a Monte
    Carlo simulator, a vanilla pricer, and an implied-volatility surface
    generator.  The base class fixes the method signatures so that the
    Fourier-pricing, calibration and risk layers can treat every model
    uniformly.
    """

    @abc.abstractmethod
    def characteristic_function(
        self,
        u: np.ndarray,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
    ) -> np.ndarray:
        """Evaluate the characteristic function of log-price at points ``u``.

        Parameters
        ----------
        u : numpy.ndarray
            Real evaluation points (frequency-domain argument).
        S0, K, T, r, q : float
            Spot, strike, maturity, risk-free rate and dividend yield.

        Returns
        -------
        numpy.ndarray
            Complex-valued characteristic function evaluated at ``u``.
        """

    @abc.abstractmethod
    def simulate(
        self,
        S0: float,
        V0: float,
        T: float,
        n_paths: int,
        n_steps: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte Carlo simulate spot and variance paths.

        Returns
        -------
        tuple of numpy.ndarray
            ``(S_paths, V_paths)`` each of shape ``(n_paths, n_steps + 1)``.
        """

    @abc.abstractmethod
    def price_vanilla(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: str = "call",
    ) -> float:
        """Price a European vanilla option."""

    @abc.abstractmethod
    def implied_vol_surface(
        self,
        S0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: float,
        q: float,
    ) -> np.ndarray:
        """Return a 2-D implied volatility surface ``(maturities x strikes)``."""

    @property
    @abc.abstractmethod
    def params(self) -> Dict[str, float]:
        """Model parameters as a dictionary."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        body = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        return f"{self.name}({body})"
