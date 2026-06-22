"""Heston (1993) characteristic function.

Two algebraically-equivalent formulations are provided:

* ``"original"`` — the formulation as it appears in Heston (1993).
* ``"little_trap"`` — the *Little Heston Trap* reorganisation of
  Albrecher, Mayer, Schoutens and Tistaert (2007) / Lord and Kahl (2010),
  which keeps the complex logarithm on its principal branch and is therefore
  numerically stable for large maturities and across the optimiser's search
  space.  This is the default.

The characteristic function returned is that of the **log spot** :math:`x_T =
\\ln S_T`, i.e.

.. math::

    \\phi(u) = \\mathbb{E}^{\\mathbb{Q}}\\!\\left[e^{i u \\ln S_T}\\right].

It is vectorised over the frequency argument ``u``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["HestonCharacteristicFunction"]

# Argument used to clip the real part of the exponent to avoid overflow when
# the CF is evaluated at large |u| during numerical integration.
_EXP_CLIP = 700.0


class HestonCharacteristicFunction:
    """Characteristic function of the Heston log-price.

    The class is stateless apart from the choice of ``formulation``; all model
    and contract parameters are passed to :meth:`cf`.  This makes it cheap to
    share a single instance across the pricing and calibration layers.

    Parameters
    ----------
    formulation : {'little_trap', 'original'}, optional
        Which algebraic representation to use.  Defaults to ``'little_trap'``.
    """

    def __init__(self, formulation: str = "little_trap") -> None:
        if formulation not in ("little_trap", "original"):
            raise ValueError(
                "formulation must be 'little_trap' or 'original', "
                f"got {formulation!r}"
            )
        self.formulation = formulation

    # ------------------------------------------------------------------
    # Core characteristic function
    # ------------------------------------------------------------------
    def cf(
        self,
        u: np.ndarray,
        S0: float,
        V0: float,
        T: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        formulation: str | None = None,
    ) -> np.ndarray:
        r"""Evaluate :math:`\phi(u) = E[e^{i u \ln S_T}]`.

        Parameters
        ----------
        u : numpy.ndarray
            Real evaluation points.
        S0, V0, T, r, q : float
            Spot, initial variance, maturity, rate, dividend yield.
        kappa, theta, xi, rho : float
            Heston parameters: mean-reversion speed, long-run variance,
            vol-of-vol and spot-vol correlation.
        formulation : str, optional
            Override the instance default for a single call.

        Returns
        -------
        numpy.ndarray
            Complex array, same shape as ``u``.

        Notes
        -----
        With :math:`F = S_0 e^{(r-q)T}` the forward, both formulations share

        .. math::

            d = \sqrt{(\rho \xi i u - \kappa)^2 + \xi^2 (i u + u^2)}.

        The *original* uses :math:`g = (b - d)/(b + d)` while the *Little Trap*
        uses :math:`g = (b + d)/(b - d)` with the sign of the exponential
        flipped, where :math:`b = \kappa - \rho \xi i u`.
        """
        form = formulation or self.formulation
        if min(theta, xi, V0) <= 0.0:
            raise ValueError(
                "theta, xi and V0 must be strictly positive in the Heston CF; "
                f"got theta={theta}, xi={xi}, V0={V0}"
            )
        # Preserve complex inputs (Lewis / Carr-Madan / MGF pass complex u).
        u = np.asarray(u) + 0j
        iu = 1j * u

        a = kappa * theta
        b = kappa - rho * xi * iu

        # Discriminant; sqrt returns the principal branch.
        d = np.sqrt(b**2 + (xi**2) * (iu + u**2))

        lnF = np.log(S0) + (r - q) * T

        if form == "little_trap":
            # Albrecher et al. (2007) stable form: g = (b - d)/(b + d), e^{-dT}.
            # Well defined at u = 0 because b + d = 2*kappa there.
            g = (b - d) / (b + d)
            exp_dt = np.exp(-d * T)
            C = (a / xi**2) * (
                (b - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
            )
            D = ((b - d) / xi**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
        else:  # original Heston (1993) "trap" form: g = (b + d)/(b - d), e^{+dT}
            g = (b + d) / (b - d)
            exp_dt = np.exp(d * T)
            C = (a / xi**2) * (
                (b + d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
            )
            D = ((b + d) / xi**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

        exponent = C + D * V0 + iu * lnF
        # Clip the real part to avoid overflow in exp at extreme u.
        re = np.clip(exponent.real, -_EXP_CLIP, _EXP_CLIP)
        return np.exp(re + 1j * exponent.imag)

    # ------------------------------------------------------------------
    # Moment generating function and log-return moments
    # ------------------------------------------------------------------
    def mgf(
        self,
        u_real: np.ndarray,
        S0: float,
        V0: float,
        T: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ) -> np.ndarray:
        r"""Moment generating function :math:`E[e^{u \ln S_T}] = \phi(-iu)`.

        Parameters
        ----------
        u_real : numpy.ndarray
            Real arguments of the MGF.

        Returns
        -------
        numpy.ndarray
            (Generally complex but real for valid inputs) MGF values.
        """
        u_real = np.asarray(u_real, dtype=np.float64)
        return self.cf(-1j * u_real, S0, V0, T, r, q, kappa, theta, xi, rho)

    def log_return_moments(
        self,
        T: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        V0: float,
    ) -> Dict[str, float]:
        r"""Analytical moments of the log return :math:`\ln(S_T/S_0)`.

        Returns
        -------
        dict
            Keys ``mean``, ``variance``, ``skewness``, ``kurtosis`` (the last
            being *excess* kurtosis).

        Notes
        -----
        The integrated-variance moments use the closed forms

        .. math::

            E\!\left[\int_0^T V_t\,dt\right]
              = \theta T + (V_0 - \theta)\frac{1 - e^{-\kappa T}}{\kappa}.

        The variance, skewness and excess kurtosis of the log return are
        obtained from the cumulants of the Heston model (Gatheral, 2006).
        """
        ekt = np.exp(-kappa * T)

        # Expected integrated variance E[ int_0^T V_t dt ].
        eiv = theta * T + (V0 - theta) * (1.0 - ekt) / kappa
        mean = (r - q) * T - 0.5 * eiv

        # Cumulants of x_T = ln S_T (Fang & Oosterlee 2008, Eq. for Heston).
        c2 = self._c2(T, kappa, theta, xi, rho, V0)
        c3 = self._c3(T, kappa, theta, xi, rho, V0)
        c4 = self._c4(T, kappa, theta, xi, rho, V0)

        variance = c2
        skewness = c3 / c2**1.5 if c2 > 0 else 0.0
        kurtosis = c4 / c2**2 if c2 > 0 else 0.0  # excess kurtosis
        return {
            "mean": float(mean),
            "variance": float(variance),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
        }

    # ------------------------------------------------------------------
    # Cumulants (used by COS truncation and moment diagnostics)
    # ------------------------------------------------------------------
    @staticmethod
    def _c2(T, kappa, theta, xi, rho, V0) -> float:
        """Second cumulant of the Heston log return (Fang-Oosterlee 2008)."""
        ekt = np.exp(-kappa * T)
        term1 = (1.0 / (8.0 * kappa**3)) * (
            xi * T * kappa * ekt * (V0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
            + kappa * rho * xi * (1.0 - ekt) * (16.0 * theta - 8.0 * V0)
            + 2.0
            * theta
            * kappa
            * T
            * (-4.0 * kappa * rho * xi + xi**2 + 4.0 * kappa**2)
            + xi**2
            * ((theta - 2.0 * V0) * ekt**2 + theta * (6.0 * ekt - 7.0) + 2.0 * V0)
            + 8.0 * kappa**2 * (V0 - theta) * (1.0 - ekt)
        )
        return float(term1)

    @staticmethod
    def _c1(T, kappa, theta, r, q, V0) -> float:
        """First cumulant (mean) of the log return."""
        ekt = np.exp(-kappa * T)
        return float(
            (r - q) * T + (1.0 - ekt) * (theta - V0) / (2.0 * kappa) - 0.5 * theta * T
        )

    @classmethod
    def _c3(cls, T, kappa, theta, xi, rho, V0) -> float:
        """Third cumulant via finite differencing of the CGF (robust)."""
        return cls._cgf_derivative(3, T, kappa, theta, xi, rho, V0)

    @classmethod
    def _c4(cls, T, kappa, theta, xi, rho, V0) -> float:
        """Fourth cumulant via finite differencing of the CGF."""
        return cls._cgf_derivative(4, T, kappa, theta, xi, rho, V0)

    @classmethod
    def _cgf_derivative(cls, n, T, kappa, theta, xi, rho, V0, h=1e-3) -> float:
        r"""``n``-th derivative at 0 of the cumulant generating function.

        The cumulant generating function of the log return is
        :math:`\psi(u) = \ln E[e^{u (\ln S_T - \ln S_0)}]`, evaluated here with
        :math:`r = q = 0` so that only the volatility-driven cumulants remain
        (the drift only affects :math:`c_1`).  A central finite-difference
        stencil gives the required derivative at :math:`u = 0`.
        """
        cf = cls(formulation="little_trap")

        def psi(u_real: float) -> complex:
            val = cf.cf(
                np.array([-1j * u_real]),
                S0=1.0,
                V0=V0,
                T=T,
                r=0.0,
                q=0.0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
            )[0]
            return np.log(val)

        if n == 3:
            # Central 5-point stencil for the third derivative.
            d = (-0.5 * psi(-2 * h) + psi(-h) - psi(h) + 0.5 * psi(2 * h)) / h**3
        elif n == 4:
            d = (
                psi(-2 * h) - 4 * psi(-h) + 6 * psi(0.0) - 4 * psi(h) + psi(2 * h)
            ) / h**4
        else:  # pragma: no cover - only 3,4 used
            raise ValueError("only n in {3, 4} supported")
        return float(np.real(d))
