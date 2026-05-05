"""Parity helpers: compare Lucid output against reference output element-wise."""

import numpy as np
import lucid
from lucid._tensor.tensor import Tensor


def parity_atol(dtype=None) -> float:
    """Return the absolute tolerance for parity comparisons.

    float16/bfloat16 → 1e-2
    float32          → 1e-4
    float64          → 1e-6
    """
    from lucid.test.helpers.numerics import tol as _tol

    if dtype is None:
        dtype = lucid.float32
    return _tol(dtype)[0]


def check_parity(
    lucid_out,
    reference_out,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    msg: str = "",
) -> None:
    """Assert that lucid_out and reference_out are element-wise close.

    Accepts Lucid Tensor plus reference tensor or plain numpy arrays.

    Parameters
    ----------
    lucid_out  : lucid.Tensor or numpy.ndarray
    reference_out : reference tensor or numpy.ndarray
    atol, rtol : tolerances
    msg        : optional failure message prefix
    """
    # Convert lucid output
    if isinstance(lucid_out, Tensor):
        l_np = lucid_out.numpy()
    else:
        l_np = np.asarray(lucid_out)

    # Convert reference output
    try:
        t_np = reference_out.detach().numpy()
    except AttributeError:
        t_np = np.asarray(reference_out)

    np.testing.assert_allclose(
        l_np,
        t_np,
        atol=atol,
        rtol=rtol,
        err_msg=f"{msg}\nLucid:\n{l_np}\nReference:\n{t_np}" if msg else None,
    )
