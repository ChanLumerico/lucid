"""
lucid.testing._comparison — numerical comparison utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def assert_close(
    actual,
    expected,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    msg: str | None = None,
) -> None:
    """Assert that *actual* and *expected* are numerically close.

    Converts both arguments to numpy arrays before comparison so they can
    be plain ndarray, Python scalar, or lucid Tensor.

    Raises
    ------
    AssertionError
        If ``|actual - expected| > atol + rtol * |expected|`` for any element.
    """
    def _to_np(x):
        if hasattr(x, "_impl"):
            return np.asarray(x._impl.data_as_python(), dtype=np.float64).reshape(x.shape)
        if hasattr(x, "numpy"):
            return np.asarray(x.numpy(), dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    a_np = _to_np(actual)
    e_np = _to_np(expected)

    if not np.allclose(a_np, e_np, atol=atol, rtol=rtol):
        diff = np.abs(a_np - e_np)
        max_diff = float(diff.max())
        suffix = f": {msg}" if msg else ""
        raise AssertionError(
            f"assert_close failed{suffix}\n"
            f"  max |actual - expected| = {max_diff:.3e}  "
            f"(atol={atol}, rtol={rtol})"
        )
