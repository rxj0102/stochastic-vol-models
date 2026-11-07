"""Unified interface to the Heston simulation schemes.

Also provides Monte Carlo vanilla pricing (with standard error) used to
cross-check the Fourier pricers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .heston_euler import HestonEulerSimulator
from .heston_exact import BroadieKayaSimulator
from .heston_milstein import HestonMilsteinSimulator
from .heston_qe import HestonQESimulator

__all__ = ["PathManager", "SimResult"]

_SCHEMES = {
    "euler": HestonEulerSimulator,
    "milstein": HestonMilsteinSimulator,
    "qe": HestonQESimulator,
    "exact": BroadieKayaSimulator,
}


@dataclass
class SimResult:
    """Container for a simulation run."""

    S_paths: np.ndarray
    V_paths: np.ndarray
    scheme: str
    elapsed_seconds: float


class PathManager:
    """Dispatch to a named simulation scheme and price by Monte Carlo."""

    def simulate(
        self,
        S0,
        V0,
        T,
        r,
        q,
        params,
        n_paths,
        n_steps,
        scheme: str = "qe",
        seed: Optional[int] = None,
        **kwargs,
    ) -> SimResult:
        """Simulate paths with the requested ``scheme``.

        Parameters
        ----------
        scheme : {'qe', 'euler', 'milstein', 'exact'}
        kwargs
            Extra keyword arguments forwarded to the scheme (e.g.
            ``antithetic=True`` for Euler).
        """
        if scheme not in _SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {list(_SCHEMES)}")
        sim = _SCHEMES[scheme]()
        t0 = time.perf_counter()
        S, V = sim.simulate(S0, V0, T, r, q, params, n_paths, n_steps, seed, **kwargs)
        return SimResult(S, V, scheme, time.perf_counter() - t0)

    def price_vanilla_mc(
        self,
        S0,
        K,
        T,
        r,
        q,
        params,
        n_paths=100_000,
        n_steps=100,
        scheme="qe",
        option_type="call",
        seed=None,
    ) -> Tuple[float, float]:
        """Monte Carlo price and standard error of a European vanilla.

        Returns
        -------
        tuple of float
            ``(price, std_error)``.
        """
        res = self.simulate(
            S0, params.V0, T, r, q, params, n_paths, n_steps, scheme, seed
        )
        ST = res.S_paths[:, -1]
        if option_type == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)
        disc = np.exp(-r * T)
        price = disc * payoff.mean()
        stderr = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))
        return float(price), float(stderr)
