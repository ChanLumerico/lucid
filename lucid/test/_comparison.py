"""
Numerical comparison utilities for testing Lucid tensor operations.

Implemented entirely with the Lucid C++ engine — no numpy dependency.
"""

from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap, _unwrap
from lucid._tensor.tensor import Tensor


def _to_tensor(x: "Tensor | list | int | float") -> Tensor:
    """Convert any scalar / list / Tensor to a Lucid Tensor on CPU."""
    if isinstance(x, Tensor):
        return x
    # Python scalar, list, or numpy array → Tensor via the interop boundary.
    from lucid._factories.converters import tensor as _tensor_fn

    return _tensor_fn(x)


def assert_close(
    actual: "Tensor | list | int | float",
    expected: "Tensor | list | int | float",
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    msg: str | None = None,
) -> None:
    """Assert that two tensors are element-wise close.

    Passes when ``|actual - expected| <= atol + rtol * |expected|``
    for every element.  Implemented entirely with the Lucid C++ engine.

    Parameters
    ----------
    actual, expected : Tensor or array_like
    atol : float   absolute tolerance (default 1e-8)
    rtol : float   relative tolerance (default 1e-5)
    msg  : str     optional prefix for the failure message

    Raises
    ------
    AssertionError on mismatch.
    """
    a_t = _to_tensor(actual)
    e_t = _to_tensor(expected)

    a = _unwrap(a_t)
    e = _unwrap(e_t)

    # Cast both to F64 for precision (no-op if already F64).
    a_f = _C_engine.astype(a, _C_engine.F64) if a.dtype != _C_engine.F64 else a
    e_f = _C_engine.astype(e, _C_engine.F64) if e.dtype != _C_engine.F64 else e

    diff = _C_engine.abs(_C_engine.sub(a_f, e_f))
    thresh = _C_engine.add(
        _C_engine.full(list(diff.shape), atol, _C_engine.F64, diff.device),
        _C_engine.mul(
            _C_engine.full(list(diff.shape), rtol, _C_engine.F64, diff.device),
            _C_engine.abs(e_f),
        ),
    )

    # all(diff <= thresh)
    ok = bool(_wrap(_C_engine.all(_C_engine.less_equal(diff, thresh))).item())
    if ok:
        return

    # ── Build a diagnostic message ──────────────────────────────────────────
    max_diff = float(_wrap(_C_engine.max(diff, [], False)).item())
    max_thresh = float(_wrap(_C_engine.max(thresh, [], False)).item())

    # Worst element (flat index)
    diff_flat = _C_engine.reshape(diff, [diff.numel()])
    worst_k = int(_wrap(_C_engine.argmax(diff_flat, 0, False)).item())

    a_flat = _C_engine.reshape(a_f, [a_f.numel()])
    e_flat = _C_engine.reshape(e_f, [e_f.numel()])
    idx_impl = _C_engine.full([1], float(worst_k), _C_engine.I32, a_flat.device)
    a_val = float(_wrap(_C_engine.gather(a_flat, idx_impl, 0)).item())
    e_val = float(_wrap(_C_engine.gather(e_flat, idx_impl, 0)).item())

    prefix = f"{msg}\n" if msg else ""
    raise AssertionError(
        f"{prefix}assert_close failed (atol={atol}, rtol={rtol}):\n"
        f"  Max |diff|     = {max_diff:.6g}\n"
        f"  Max threshold  = {max_thresh:.6g}\n"
        f"  Worst element  [flat {worst_k}]: actual={a_val:.6g}, expected={e_val:.6g}, "
        f"|diff|={abs(a_val - e_val):.6g}"
    )
