"""
Numerical gradient checker using finite differences.
"""

from typing import Any, Callable, Sequence

import numpy as np


def gradcheck(
    func: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Compare analytical gradients from backward() against finite-difference Jacobians.

    Parameters
    ----------
    func : callable
        Function mapping Tensor inputs to a scalar Tensor output.
    inputs : sequence of Tensor
        Inputs to func that require grad. All must be float tensors.
    eps : float
        Finite-difference step size.
    atol : float
        Absolute tolerance for the comparison.
    rtol : float
        Relative tolerance for the comparison.
    raise_exception : bool
        If True (default), raise AssertionError on mismatch. Otherwise return False.

    Returns
    -------
    bool
        True if all analytical and numerical gradients agree within tolerance.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import gradcheck
    >>> x = lucid.randn(3, requires_grad=True)
    >>> gradcheck(lambda t: t.sum(), [x])
    True
    """
    from lucid._C import engine as _C_engine

    def _to_numpy(t: Any) -> np.ndarray:  # type: ignore[type-arg]
        if hasattr(t, "_impl"):
            return np.array(t._impl.data_as_python(), dtype=np.float64)
        return np.asarray(t, dtype=np.float64)

    def _clone_with_grad(t: Any) -> Any:
        from lucid._factories.converters import tensor as _tensor_fn

        arr = _to_numpy(t)
        return _tensor_fn(arr.copy(), requires_grad=True)

    # Clone inputs so we don't mutate originals
    inputs_clone = [_clone_with_grad(t) for t in inputs]

    # ── Analytical gradients ─────────────────────────────────────────────────
    out = func(*inputs_clone)
    if out._impl.numel() != 1:
        raise ValueError(
            "gradcheck requires a scalar-valued function output "
            f"(got shape {tuple(out._impl.shape)})"
        )
    out.backward()
    analytical = [_to_numpy(t.grad) for t in inputs_clone]

    # ── Numerical gradients (central differences) ────────────────────────────
    numerical = []
    for inp in inputs:
        arr = _to_numpy(inp)
        grad_num = np.zeros_like(arr, dtype=np.float64)
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = float(arr[idx])

            arr[idx] = orig + eps
            from lucid._factories.converters import tensor as _tensor_fn

            inp_plus = _tensor_fn(arr.copy())
            inp_list = []
            for other in inputs:
                if other is inp:
                    inp_list.append(inp_plus)
                else:
                    inp_list.append(_tensor_fn(_to_numpy(other).copy()))
            f_plus = float(_to_numpy(func(*inp_list)).flat[0])

            arr[idx] = orig - eps
            inp_minus = _tensor_fn(arr.copy())
            inp_list2 = []
            for other in inputs:
                if other is inp:
                    inp_list2.append(inp_minus)
                else:
                    inp_list2.append(_tensor_fn(_to_numpy(other).copy()))
            f_minus = float(_to_numpy(func(*inp_list2)).flat[0])

            grad_num[idx] = (f_plus - f_minus) / (2 * eps)
            arr[idx] = orig
            it.iternext()
        numerical.append(grad_num)

    # ── Compare ───────────────────────────────────────────────────────────────
    for i, (an, nu) in enumerate(zip(analytical, numerical)):
        try:
            np.testing.assert_allclose(an, nu, atol=atol, rtol=rtol)
        except AssertionError as exc:
            msg = f"Gradient check failed for input {i}:\n{exc}"
            if raise_exception:
                raise AssertionError(msg) from None
            return False
    return True
