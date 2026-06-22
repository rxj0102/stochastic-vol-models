"""Multi-stage optimisation for model calibration.

A global differential-evolution search is followed by a local L-BFGS-B
refinement warm-started from the global solution — the standard recipe for the
multi-modal Heston calibration objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize

__all__ = ["CalibrationOptimizer", "OptimizeOutcome", "DEFAULT_BOUNDS"]

# (kappa, theta, xi, rho, V0)
DEFAULT_BOUNDS: List[Tuple[float, float]] = [
    (0.01, 10.0),
    (0.001, 1.0),
    (0.01, 2.0),
    (-0.99, 0.99),
    (0.001, 1.0),
]


@dataclass
class OptimizeOutcome:
    """Result of a calibration optimisation."""

    x: np.ndarray
    fun: float
    converged: bool
    n_iter: int
    history: List[float] = field(default_factory=list)


class CalibrationOptimizer:
    """Two-stage (global + local) optimiser for calibration objectives."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    def optimize(
        self,
        loss_fn: Callable[[Sequence[float]], float],
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "two_stage",
        x0: Optional[np.ndarray] = None,
        de_maxiter: int = 200,
    ) -> OptimizeOutcome:
        """Minimise ``loss_fn`` over ``bounds``.

        Parameters
        ----------
        loss_fn : callable
            Maps a parameter vector to a scalar loss.
        bounds : list of (low, high), optional
            Defaults to :data:`DEFAULT_BOUNDS`.
        method : {'two_stage', 'global', 'local'}
        x0 : numpy.ndarray, optional
            Starting point for ``'local'`` (required) or seed for refinement.
        de_maxiter : int, optional
            Maximum differential-evolution generations.
        """
        bounds = bounds or DEFAULT_BOUNDS
        history: List[float] = []

        def tracked(x):
            val = loss_fn(x)
            history.append(val)
            return val

        if method == "local":
            if x0 is None:
                raise ValueError("method='local' requires x0")
            res = minimize(
                tracked,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500},
            )
            return OptimizeOutcome(
                res.x, float(res.fun), bool(res.success), res.nit, history
            )

        # Stage 1: global search.
        de = differential_evolution(
            tracked,
            bounds,
            maxiter=de_maxiter,
            tol=1e-7,
            strategy="best1bin",
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.9,
            seed=self.seed,
            polish=False,
            init="latinhypercube",
            x0=x0,
        )
        if method == "global":
            return OptimizeOutcome(
                de.x, float(de.fun), bool(de.success), de.nit, history
            )

        # Stage 2: local refinement warm-started from DE.
        res = minimize(
            tracked,
            de.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )
        best_x, best_f = (res.x, res.fun) if res.fun <= de.fun else (de.x, de.fun)
        return OptimizeOutcome(
            best_x,
            float(best_f),
            bool(res.success or de.success),
            de.nit + res.nit,
            history,
        )

    def calibrate_with_restarts(
        self,
        loss_fn: Callable,
        bounds: Optional[List[Tuple[float, float]]] = None,
        n_restarts: int = 5,
    ) -> OptimizeOutcome:
        """Run several local optimisations from random starts; keep the best."""
        bounds = bounds or DEFAULT_BOUNDS
        rng = np.random.default_rng(self.seed)
        best: Optional[OptimizeOutcome] = None
        for _ in range(n_restarts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            out = self.optimize(loss_fn, bounds, method="local", x0=x0)
            if best is None or out.fun < best.fun:
                best = out
        return best
