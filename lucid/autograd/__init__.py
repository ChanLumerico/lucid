from typing import Iterable, Sequence
import weakref

import numpy as np

from lucid._backend.metal import mx
from lucid.error import BackwardError
from lucid.types import _MLXArray, _NumPyArray, _TensorLike, _Scalar, _Gradient


__all__ = ["grad", "backward"]


def _as_tuple(value: _TensorLike | Sequence[_TensorLike]) -> tuple[_TensorLike, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _coerce_grad_output(
    output: _TensorLike, grad_output: _TensorLike | _Gradient | _Scalar
) -> _Gradient:
    if grad_output is None:
        if output.is_cpu():
            return np.ones_like(output.data)
        return mx.ones_like(output.data)

    if isinstance(grad_output, _TensorLike):
        grad_output = grad_output.data

    if isinstance(grad_output, _NumPyArray):
        if output.is_gpu():
            grad_output = mx.array(grad_output)
        return grad_output

    if isinstance(grad_output, _MLXArray):
        if output.is_cpu():
            grad_output = np.array(grad_output)
        return grad_output

    if output.is_cpu():
        return np.ones_like(output.data) * grad_output
    return mx.ones_like(output.data) * grad_output


def grad(
    outputs: _TensorLike | Iterable[_TensorLike],
    inputs: _TensorLike | Iterable[_TensorLike],
    grad_outputs: _TensorLike | Iterable[_TensorLike] | Iterable[_Scalar] | None = None,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> tuple[_Gradient, ...] | _Gradient:
    out_tensors = _as_tuple(outputs)
    in_tensors = _as_tuple(inputs)

    if grad_outputs is None:
        grad_outs = (None,) * len(out_tensors)
    else:
        grad_outs = _as_tuple(grad_outputs)
        if len(grad_outs) != len(out_tensors):
            raise ValueError("grad_outputs length must match outputs length.")

    for tensor in out_tensors:
        if not isinstance(tensor, _TensorLike):
            raise TypeError("All outputs must be _TensorLike instances.")

    for tensor in in_tensors:
        if not isinstance(tensor, _TensorLike):
            raise TypeError("All inputs must be _TensorLike instances.")

    prev_grads = {tensor: tensor.grad for tensor in in_tensors}
    prev_out_grads = {tensor: tensor.grad for tensor in out_tensors}
    prev_keep = {tensor: tensor.keep_grad for tensor in in_tensors}

    for tensor in in_tensors:
        tensor.grad = None
        tensor.keep_grad = True

    try:
        for i, (output, grad_output) in enumerate(zip(out_tensors, grad_outs)):
            if not output.requires_grad:
                if allow_unused:
                    continue
                raise RuntimeError("All outputs must require gradients.")

            coerced = _coerce_grad_output(output, grad_output)
            if coerced.shape != output.shape:
                raise ValueError("grad_output shape must match output shape.")

            output.grad = coerced
            output.backward(
                retain_grad=True,
                retain_graph=(i < len(out_tensors) - 1) or retain_graph,
            )

        grads = tuple(tensor.grad for tensor in in_tensors)
        if not allow_unused and any(grad is None for grad in grads):
            raise RuntimeError(
                "Some inputs did not receive gradients. Set allow_unused=True."
            )

        if len(grads) == 1:
            return grads[0]
        return grads
    finally:
        for tensor, grad in prev_out_grads.items():
            tensor.grad = grad
        for tensor in in_tensors:
            tensor.grad = prev_grads[tensor]
            tensor.keep_grad = prev_keep[tensor]


def backward(
    tensor: _TensorLike, retain_grad: bool = False, retain_graph: bool = False
) -> None:
    if tensor.grad is None:
        tensor.grad = (
            np.ones_like(tensor.data) if tensor.is_cpu() else mx.ones_like(tensor.data)
        )

    visited = set()
    topo_order: list[_TensorLike] = []
    stack = [tensor]
    ops_to_clear = set()

    while stack:
        node = stack[-1]
        if node in visited:
            stack.pop()
            topo_order.append(node)
            continue

        visited.add(node)
        for parent in node._prev:
            if parent not in visited:
                stack.append(parent)

    from lucid._fusion import ENABLE_FUSION

    if ENABLE_FUSION and tensor.is_cpu():
        _try_backward_fusion(topo_order)

    for node in reversed(topo_order):
        try:
            node._backward_op()
        except Exception as e:
            raise BackwardError(shape=node.shape, op=node._op) from e

        for hook in node._backward_hooks:
            hook(node, node.grad)

        if node._op is not None:
            ops_to_clear.add(node._op)

        if not (node.is_leaf or retain_grad or node.keep_grad):
            node.grad = None

    if not retain_graph:
        for node in topo_order:
            node.clear_node()
        for op in ops_to_clear:
            try:
                op.clear()
            except Exception:
                try:
                    op.result = None
                except Exception:
                    pass


def _try_backward_fusion(topo_order: list[_TensorLike]) -> None:
    from lucid._fusion import match_fusion_table

    consumer_of: dict[int, _TensorLike] = {}
    multi_consumer: set[int] = set()

    for consumer in topo_order:
        for parent in consumer._prev:
            pid = id(parent)
            if pid in multi_consumer:
                continue

            prev_consumer = consumer_of.get(pid)
            if prev_consumer is None:
                consumer_of[pid] = consumer
            else:
                multi_consumer.add(pid)
                consumer_of.pop(pid, None)

    if not consumer_of:
        return

    for pid, v in list(consumer_of.items()):
        p = next((t for t in v._prev if id(t) == pid), None)
        if p is None:
            continue
        if p._op is None or v._op is None:
            continue

        fused_backward_op = match_fusion_table(p._op, v._op)
        if fused_backward_op is None:
            continue
        if v.size < fused_backward_op.heuristic_thresh:
            continue

        if len(v._prev) != 1 or v._prev[0] is not p:
            continue

        p_parents = tuple(p._prev)
        v._prev.remove(p)
        v._prev.extend(p_parents)
        p.clear_node(clear_op=False)

        v._backward_op.override_tensor_refs(tuple(weakref.ref(t) for t in v._prev))
        v._backward_op.override_grad_func(
            fused_backward_op.get_fused_grad_func(
                inputs=p_parents, results=v, device=v.device
            )
        )
