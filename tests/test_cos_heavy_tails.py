"""Regression test: COS pricing with heavy tails and long maturities.

With xi = 1 and rho = -0.9 the truncation range is wide. Pricing calls directly blew up
(an at-the-money 3 year call came out near -1.7e7); calls are now priced as puts plus
put-call parity, with the number of terms raised until the price converges.
Reference values agree with the Carr-Madan and Lewis pricers in this package.
"""
import numpy as np
import pytest

from fourier_pricing.carr_madan import CarrMadanPricer
from fourier_pricing.cos_method import COSMethodPricer
from fourier_pricing.lewis_method import LewisMethodPricer
from models.heston import HestonParams

HEAVY = HestonParams(kappa=0.5, theta=0.09, xi=1.0, rho=-0.9, V0=0.09)


@pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
@pytest.mark.parametrize("T", [1.0, 3.0])
def test_cos_matches_lewis_and_carr_madan(K, T):
    cos = COSMethodPricer().price(100.0, K, T, 0.03, 0.01, HEAVY, "call")
    lewis = LewisMethodPricer().price(100.0, K, T, 0.03, 0.01, HEAVY, "call")
    cm = CarrMadanPricer().price(100.0, K, T, 0.03, 0.01, HEAVY, "call")
    assert lewis == pytest.approx(cm, abs=1e-4)
    assert cos == pytest.approx(lewis, abs=1e-4)


def test_cos_put_call_parity():
    f = COSMethodPricer()
    c = f.price(100.0, 100.0, 3.0, 0.03, 0.01, HEAVY, "call")
    p = f.price(100.0, 100.0, 3.0, 0.03, 0.01, HEAVY, "put")
    assert c - p == pytest.approx(100.0 * np.exp(-0.01 * 3.0) - 100.0 * np.exp(-0.03 * 3.0), abs=1e-10)
