from typing import Any, Callable, Sequence

import numpy as np

import lucid

from lucid.test.parity.core import Input, TensorInput, ScalarInput


def _wrap_for_eval(
    inputs: Sequence[Input], overrides: dict[int, np.ndarray]
) -> list[Any]:
    wrapped: list[Any] = []
    for i, item in enumerate(inputs):
        if isinstance(item, TensorInput):
            arr = overrides.get(i, item.array).copy()
            kwargs: dict[str, Any] = {"requires_grad": False}
            if item.dtype_override is not None:
                kwargs["dtype"] = item.dtype_override
            wrapped.append(lucid.tensor(arr, **kwargs))
        elif isinstance(item, ScalarInput):
            wrapped.append(item.value)
        else:
            wrapped.append(item)
    return wrapped


def _as_scalar_loss(out: Any) -> Any:
    if hasattr(out, "ndim") and out.ndim == 0:
        return out
    return out.sum()


def numerical_jacobian(
    fn: Callable[..., Any], inputs: Sequence[Input], *, eps: float = 1e-06
) -> list[np.ndarray]:
    grads: list[np.ndarray | None] = []
    for idx, item in enumerate(inputs):
        if not (isinstance(item, TensorInput) and item.requires_grad):
            grads.append(None)
            continue
        base = item.array.astype(np.float64)
        grad = np.zeros_like(base)
        it = np.nditer(base, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            multi = it.multi_index
            orig = base[multi]
            plus = base.copy()
            plus[multi] = orig + eps
            out_plus = _as_scalar_loss(fn(*_wrap_for_eval(inputs, {idx: plus})))
            minus = base.copy()
            minus[multi] = orig - eps
            out_minus = _as_scalar_loss(fn(*_wrap_for_eval(inputs, {idx: minus})))
            v_plus = float(
                np.asarray(out_plus.data if hasattr(out_plus, "data") else out_plus)
            )
            v_minus = float(
                np.asarray(out_minus.data if hasattr(out_minus, "data") else out_minus)
            )
            grad[multi] = (v_plus - v_minus) / (2 * eps)
            it.iternext()
        grads.append(grad)
    return grads


def assert_gradcheck(
    fn: Callable[..., Any],
    inputs: Sequence[Input],
    *,
    eps: float = 1e-06,
    rtol: float = 0.001,
    atol: float = 0.0001,
) -> None:
    analytical_inputs: list[Any] = []
    for item in inputs:
        if isinstance(item, TensorInput):
            kwargs: dict[str, Any] = {"requires_grad": item.requires_grad}
            if item.dtype_override is not None:
                kwargs["dtype"] = item.dtype_override
                arr = item.array.copy()
            else:
                arr = item.array.astype(np.float64)
            analytical_inputs.append(lucid.tensor(arr, **kwargs))
        elif isinstance(item, ScalarInput):
            analytical_inputs.append(item.value)
        else:
            analytical_inputs.append(item)
    out = fn(*analytical_inputs)
    loss = _as_scalar_loss(out)
    loss.backward()
    numeric = numerical_jacobian(fn, inputs, eps=eps)
    for idx, (item, num) in enumerate(zip(inputs, numeric)):
        if num is None:
            continue
        ana = analytical_inputs[idx].grad
        assert ana is not None, f"gradcheck: input[{idx}] analytical grad is None"
        ana_arr = np.asarray(ana).astype(np.float64)
        try:
            np.testing.assert_allclose(ana_arr, num, rtol=rtol, atol=atol)
        except AssertionError as err:
            raise AssertionError(f"gradcheck[input#{idx}] mismatch: {err}") from None
