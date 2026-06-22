"""Shared fixtures for the test suite."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.heston import HestonParams  # noqa: E402


@pytest.fixture
def base_params() -> HestonParams:
    """A well-behaved, Feller-satisfying Heston parameter set."""
    return HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, V0=0.04)


@pytest.fixture
def market():
    """Standard contract environment."""
    return {"S0": 100.0, "r": 0.02, "q": 0.0, "T": 1.0}
