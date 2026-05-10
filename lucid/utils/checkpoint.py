"""
Gradient checkpointing: trade memory for recomputation during backward.
"""

from typing import Callable, cast
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function, FunctionCtx
from lucid.autograd._grad_mode import no_grad, enable_grad
import lucid.autograd as _autograd


def checkpoint(
    function: Callable[..., Tensor | tuple[Tensor, ...]],
    *args: Tensor,
    use_reentrant: bool = False,
    **kwargs: object,
) -> Tensor | tuple[Tensor, ...]:
    """Run function without saving intermediate activations; recompute during backward."""

    class CheckpointFunction(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
            ctx.function = function
            ctx.kwargs = kwargs
            ctx.num_inputs = len(inputs)
            ctx.save_for_backward(*inputs)
            with no_grad():
                return function(*inputs, **kwargs)

        @staticmethod
        def backward(
            ctx: FunctionCtx, *grad_outputs: Tensor
        ) -> Tensor | tuple[Tensor, ...]:
            saved = ctx.saved_tensors
            inputs_detached = [
                s.detach().requires_grad_(s.requires_grad) for s in saved
            ]
            fn: Callable[..., Tensor | tuple[Tensor, ...]] = cast(
                Callable[..., Tensor | tuple[Tensor, ...]], ctx.function
            )
            kw: dict[str, object] = cast(dict[str, object], ctx.kwargs)
            with enable_grad():
                output = fn(*inputs_detached, **kw)

            if not isinstance(output, (list, tuple)):
                output = (output,)
            grads = _autograd.grad(
                list(output),
                [inp for inp in inputs_detached if inp.requires_grad],
                grad_outputs=list(grad_outputs),
                allow_unused=True,
            )

            grad_iter = iter(grads)
            result: tuple[Tensor | None, ...] = tuple(
                next(grad_iter) if inp.requires_grad else None
                for inp in inputs_detached
            )
            return result  # type: ignore[return-value]  # tuple[Tensor|None,...] is valid backward return

    return CheckpointFunction.apply(*args)  # type: ignore[return-value]
