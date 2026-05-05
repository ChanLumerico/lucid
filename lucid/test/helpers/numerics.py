"""
Numeric helpers for the lucid test suite.

make_tensor     — Create a deterministic random Lucid tensor.
tol             — Return (atol, rtol) appropriate for a dtype.
rand_like_reference — Wrap reference tensor data as a Lucid tensor.
"""

import numpy as np
import lucid
from lucid._tensor.tensor import Tensor


def make_tensor(
    shape: tuple | list,
    dtype=None,
    device: str = "cpu",
    low: float = -1.0,
    high: float = 1.0,
    seed: int = 0,
    requires_grad: bool = False,
) -> Tensor:
    """Create a deterministic random tensor with uniform values in [low, high].

    Parameters
    ----------
    shape   : tensor shape
    dtype   : lucid dtype (default: lucid.float32)
    device  : "cpu" or "gpu"
    low/high: value range
    seed    : RNG seed for reproducibility
    requires_grad: whether to enable gradient tracking
    """
    if dtype is None:
        dtype = lucid.float32
    rng = np.random.default_rng(seed)

    # Use numpy float32 as intermediate regardless of target dtype
    data = rng.uniform(low, high, shape).astype(np.float32)
    t = lucid.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    return t


def make_int_tensor(
    shape: tuple | list,
    low: int = 0,
    high: int = 10,
    dtype=None,
    device: str = "cpu",
    seed: int = 0,
) -> Tensor:
    """Create a deterministic random integer tensor."""
    if dtype is None:
        dtype = lucid.int64
    rng = np.random.default_rng(seed)
    data = rng.integers(low, high, shape).astype(np.int64)
    return lucid.tensor(data, dtype=dtype, device=device)


def tol(dtype) -> tuple[float, float]:
    """Return (atol, rtol) tolerances appropriate for a Lucid dtype.

    Returns
    -------
    (atol, rtol) as floats
    """
    _MAP = {
        lucid.float16: (1e-2, 1e-2),
        lucid.float32: (1e-4, 1e-4),
        lucid.float64: (1e-7, 1e-7),
        lucid.bfloat16: (1e-2, 1e-2),
    }
    return _MAP.get(dtype, (1e-4, 1e-4))


def rand_like_reference(reference_tensor) -> Tensor:
    """Wrap reference tensor numpy data as a Lucid Tensor.

    Used in parity tests to ensure both sides use identical data.

    Parameters
    ----------
    reference_tensor : array-like object with detach().numpy()

    Returns
    -------
    Lucid Tensor with same data
    """
    data = reference_tensor.detach().numpy()
    return lucid.tensor(data.copy())
