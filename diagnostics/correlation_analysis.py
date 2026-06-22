"""How the spot-vol correlation rho shapes the Heston smile."""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["CorrelationEffects"]


class CorrelationEffects:
    """Quantify the effect of ``rho`` on smile skew and the leverage effect."""

    def rho_smile_sensitivity(
        self, params_base, rho_range, S0, K_range, T, r, q
    ) -> Dict[str, np.ndarray]:
        """Family of smiles as ``rho`` is swept, holding other params fixed."""
        from dataclasses import replace

        from implied_vol.surface import IVSurface

        K_range = np.asarray(K_range, dtype=float)
        smiles = []
        for rho in rho_range:
            p = replace(params_base, rho=float(rho))
            surf = IVSurface.from_heston(p, S0, r, q, K_range, np.array([T]))
            smiles.append(surf.iv[0])
        return {
            "rho": np.asarray(rho_range),
            "strikes": K_range,
            "smiles": np.array(smiles),
        }

    def decompose_skew(self, params, S0, maturities, r, q) -> Dict[str, np.ndarray]:
        """ATM skew per maturity and its rho/xi attribution proxies."""
        from implied_vol.smile_analytics import SmileAnalytics

        sa = SmileAnalytics()
        return {"decomposition": sa.skew_decomposition(params, maturities)}

    @staticmethod
    def leverage_effect_from_paths(S_paths: np.ndarray, V_paths: np.ndarray) -> float:
        r"""Empirical correlation of spot returns and variance increments.

        Should recover the Heston ``rho`` to within Monte Carlo error.
        """
        dlogS = np.diff(np.log(S_paths), axis=1).ravel()
        dV = np.diff(V_paths, axis=1).ravel()
        mask = np.isfinite(dlogS) & np.isfinite(dV)
        if mask.sum() < 2:
            return float("nan")
        return float(np.corrcoef(dlogS[mask], dV[mask])[0, 1])
