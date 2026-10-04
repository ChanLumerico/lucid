"""
Bridge between Python autograd.Function and the C++ engine's backward graph.
"""

from collections.abc import Sequence
from typing import NamedTuple, Protocol

import lucid
from lucid._C import engine as _C_engine
from lucid._device import device as _Device
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor
from lucid.dtypes import dtype as _DType


class _FunctionClass(Protocol):
    @classmethod
    def backward(
        cls, ctx: object, *grad_outputs: Tensor
    ) -> Tensor | tuple[Tensor | None, ...] | list[Tensor | None]: ...


class _InputMeta(NamedTuple):
    """What a gradient for one tensor input has to look like.

    Taken when ``forward`` returns, as the reference framework takes its
    input metadata, so ``backward`` is held to the input it was given —
    and only the description is kept, not the tensor, so registering a
    node does not keep its inputs alive.
    """

    shape: tuple[int, ...]
    dtype: _DType
    is_complex: bool
    device: _Device
    requires_grad: bool


def _expandable(shape: tuple[int, ...], target: tuple[int, ...]) -> bool:
    """Whether ``shape`` broadcasts to ``target`` — the input to its gradient."""
    if len(shape) > len(target):
        return False
    for have, want in zip(reversed(shape), reversed(target)):
        if have != want and have != 1:
            return False
    return True


def _sum_to(grad: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reduce a gradient for a broadcast input back to the input's shape."""
    lead = grad.ndim - len(shape)
    if lead > 0:
        grad = grad.sum(dim=tuple(range(lead)))
    axes = tuple(
        i for i, size in enumerate(shape) if size == 1 and grad.shape[i] != 1
    )
    if axes:
        grad = grad.sum(dim=axes, keepdim=True)
    return grad


def _per_input(
    name: str,
    grads: list[object],
    is_tensor_arg: tuple[bool, ...],
    n_tensors: int,
) -> list[object]:
    """Line ``backward``'s results up with the node's tensor inputs.

    ``backward`` may answer in either of two ways.  One gradient per
    positional argument of ``forward`` is the documented form — ``None``
    for each argument that is not a tensor — and trailing ``None`` entries
    beyond that are ignored, as the reference framework ignores them.
    One gradient per *tensor* argument is what every built-in ``Function``
    returns and is accepted as well; the two coincide when every argument
    is a tensor.

    Raises
    ------
    RuntimeError
        For any other count, or a gradient for an argument that is not a
        tensor.
    """
    n_args = len(is_tensor_arg)
    if len(grads) > n_args and all(g is None for g in grads[n_args:]):
        grads = grads[:n_args]
    if len(grads) == n_args:
        picked: list[object] = []
        for i, (g, is_tensor) in enumerate(zip(grads, is_tensor_arg)):
            if is_tensor:
                picked.append(g)
            elif g is not None:
                raise RuntimeError(
                    f"function {name} returned a gradient different than None "
                    f"at position {i + 1}, but the corresponding forward input "
                    "was not a Tensor"
                )
        return picked
    if len(grads) == n_tensors:
        return grads
    expected = (
        f"{n_args}"
        if n_args == n_tensors
        else f"{n_args}, one per positional input, or {n_tensors}, one per tensor input"
    )
    raise RuntimeError(
        f"function {name} returned an incorrect number of gradients "
        f"(expected {expected}, got {len(grads)})"
    )


def _validate(name: str, index: int, grad: Tensor, meta: _InputMeta) -> Tensor:
    """Hold one gradient to its input's shape, dtype and device.

    Follows the reference framework's check on every node's results:

    * a gradient of the input's shape passes; one the input broadcasts to
      is summed back down to it; anything else is refused;
    * a dtype that differs is cast to the input's — a complex gradient for
      a real input keeps its real part, the convention the leaf
      accumulator already follows;
    * a device that differs is refused, except for a 0-d gradient, which
      is moved.

    The engine reads the returned buffer with the input's shape and dtype,
    so before this check a gradient of the wrong size was read short or
    past its end, a wrong dtype was reinterpreted bit for bit wherever the
    input was not a leaf, and a gradient on the wrong device became a
    ``.grad`` every read of which raised ``bad_variant_access``.
    """
    got = tuple(grad.shape)
    if got != meta.shape:
        if not _expandable(meta.shape, got):
            raise RuntimeError(
                f"Function {name} returned an invalid gradient at index {index} - "
                f"got {list(got)} but expected shape compatible with "
                f"{list(meta.shape)}"
            )
        grad = _sum_to(grad, meta.shape)
    if grad.dtype != meta.dtype:
        if grad.is_complex() and not meta.is_complex:
            grad = lucid.real(grad)
        if grad.dtype != meta.dtype:
            grad = grad.to(meta.dtype)
    if grad.device != meta.device:
        if grad.ndim != 0:
            raise RuntimeError(
                f"Function {name} returned an invalid gradient at index {index} - "
                f"expected device {meta.device.type} but got {grad.device.type}"
            )
        grad = grad.to(meta.device)
    return grad


def _register(
    outputs: Tensor | tuple[Tensor, ...],
    fn_class: _FunctionClass,
    py_ctx: object,
    tensor_inputs: list[Tensor],
    is_tensor_arg: Sequence[bool] | None = None,
) -> None:
    """
    Wire a Python backward function into the C++ autograd graph.

    The C++ engine calls backward_fn(cpp_ctx, *grad_impls) during backprop —
    one gradient per forward output. We use a closure to capture py_ctx and
    the user's backward() method.

    A ``forward`` returning several tensors registers them all against this
    one node. Their gradients come back at different points in the traversal,
    so the node runs as a barrier on the C++ side and calls back exactly once
    with the complete set; outputs the loss never used arrive as zeros.

    ``is_tensor_arg`` marks which of ``forward``'s positional arguments were
    tensors, so ``backward`` may return one gradient per argument; it
    defaults to every argument being one of ``tensor_inputs``.  Each
    gradient ``backward`` returns is checked against its input — shape,
    dtype, device — before the engine sees it, in the eager, barrier and
    ``create_graph`` paths alike, since all three call ``backward_fn``.
    """
    # C++ FunctionCtx — used as a carrier; C++ engine needs it to be non-null
    cpp_ctx = _C_engine.FunctionCtx()

    name = f"{getattr(fn_class, '__name__', 'Function')}Backward"
    metas = [
        _InputMeta(
            tuple(t.shape), t.dtype, t.is_complex(), t.device, t.requires_grad
        )
        for t in tensor_inputs
    ]
    positional = (
        tuple(bool(b) for b in is_tensor_arg)
        if is_tensor_arg is not None
        else (True,) * len(tensor_inputs)
    )

    def backward_fn(
        _cpp_ctx: object, *grad_impls: _C_engine.TensorImpl
    ) -> list[_C_engine.TensorImpl | None]:
        """Called by C++ engine: (cpp_ctx, *grad_impls) → list[TensorImpl | None]"""
        grads = fn_class.backward(py_ctx, *(_wrap(g) for g in grad_impls))
        returned: list[object] = (
            list(grads) if isinstance(grads, (list, tuple)) else [grads]
        )

        result: list[_C_engine.TensorImpl | None] = []
        for i, (g, meta) in enumerate(
            zip(_per_input(name, returned, positional, len(metas)), metas)
        ):
            # An input that needs no gradient has no edge to deliver one to;
            # like the reference, its result is neither checked nor kept.
            if not isinstance(g, Tensor) or not meta.requires_grad:
                result.append(None)
                continue
            result.append(_unwrap(_validate(name, i, g, meta)))
        return result

    node = _C_engine._PythonBackwardNode()
    node.ctx = cpp_ctx
    node.backward_fn = backward_fn
    # A backward that leaves the graph says so, and create_graph=True through
    # it is refused instead of answered with a detached gradient.
    node.once_differentiable = bool(getattr(fn_class, "_once_differentiable", False))

    impl_inputs = [_unwrap(t) for t in tensor_inputs]
    out_seq = outputs if isinstance(outputs, tuple) else (outputs,)
    _C_engine._register_python_backward_node(
        [t._impl for t in out_seq], node, impl_inputs
    )
