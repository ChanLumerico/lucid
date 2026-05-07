"""
Gradient checkpointing: trade memory for recomputation during backward.
"""

from typing import Callable, TYPE_CHECKING
from lucid._tensor.tensor import Tensor
from lucid.autograd.function import Function
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
        def forward(ctx: object, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
            ctx.function = function
            ctx.kwargs = kwargs
            ctx.num_inputs = len(inputs)
            ctx.save_for_backward(*inputs)
            with no_grad():
                return function(*inputs, **kwargs)

        @staticmethod
        def backward(ctx: object, *grad_outputs: Tensor) -> tuple[Tensor | None, ...]:
            saved = ctx.saved_tensors
            inputs_detached = [
                s.detach().requires_grad_(s.requires_grad) for s in saved
            ]
            with enable_grad():
                output = ctx.function(*inputs_detached, **ctx.kwargs)

            if not isinstance(output, (list, tuple)):
                output = (output,)
            grads = _autograd.grad(
                list(output),
                [inp for inp in inputs_detached if inp.requires_grad],
                grad_outputs=list(grad_outputs),
                allow_unused=True,
            )

            grad_iter = iter(grads)
            return tuple(
                next(grad_iter) if inp.requires_grad else None
                for inp in inputs_detached
            )

    return CheckpointFunction.apply(*args)
