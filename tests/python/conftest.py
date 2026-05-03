"""
Shared pytest fixtures and configuration for Python layer tests.
"""

import pytest
import numpy as np
import lucid


@pytest.fixture(autouse=True)
def reset_grad_mode():
    """Ensure gradient mode is enabled before each test."""
    lucid.autograd.set_grad_enabled(True)
    yield
    lucid.autograd.set_grad_enabled(True)


@pytest.fixture
def seed():
    """Fix numpy random seed for reproducibility."""
    np.random.seed(42)
    yield 42


def assert_close(a: np.ndarray, b: np.ndarray, rtol: float = 1e-4, atol: float = 1e-6) -> None:
    """Assert two arrays are close within tolerances."""
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
