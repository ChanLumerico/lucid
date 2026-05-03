"""
Numerical comparison utilities for testing Lucid tensor operations.
"""

import numpy as np
from lucid._tensor.tensor import Tensor


def assert_close(
    actual: Tensor | np.ndarray | list[object] | int | float,
    expected: Tensor | np.ndarray | list[object] | int | float,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    msg: str | None = None,
) -> None:
    """Assert that two tensors (or arrays) are element-wise close.

    Analogous to ``torch.testing.assert_close``. Passes when:
    ``|actual - expected| <= atol + rtol * |expected|`` for all elements.

    Parameters
    ----------
    actual : Tensor or array_like
        Computed value.
    expected : Tensor or array_like
        Reference value.
    atol : float, optional
        Absolute tolerance (default: 1e-8).
    rtol : float, optional
        Relative tolerance (default: 1e-5).
    msg : str, optional
        Custom failure message prepended to the diff report.

    Raises
    ------
    AssertionError
        When the tensors are not close within the given tolerances.

    Examples
    --------
    >>> from lucid.testing import assert_close
    >>> import lucid
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> assert_close(x, x.clone())
    """

    def _to_numpy(t: Tensor | np.ndarray | list[object] | int | float) -> np.ndarray:  # type: ignore[type-arg]
        if hasattr(t, "numpy"):
            return np.asarray(t.numpy())
        if hasattr(t, "_impl"):
            return np.asarray(t._impl.data_as_python())
        return np.asarray(t)

    a = _to_numpy(actual)
    e = _to_numpy(expected)

    prefix = f"{msg}\n" if msg else ""
    try:
        np.testing.assert_allclose(a, e, atol=atol, rtol=rtol, err_msg=prefix)
    except AssertionError as exc:
        raise AssertionError(
            f"{prefix}assert_close failed (atol={atol}, rtol={rtol}):\n{exc}"
        ) from None
