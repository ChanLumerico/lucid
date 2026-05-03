"""
Gradient checkpointing: trade memory for recomputation during backward.
"""

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def checkpoint(
    function: Callable[..., Any],
    *args: Any,
    use_reentrant: bool = False,
    **kwargs: Any,
) -> Any:
    """Run function without saving intermediate activations; recompute during backward.

    Reduces memory usage at the cost of extra forward computation in the backward pass.
    The function must only take Tensors as inputs/outputs.

    Args:
        function:      The function to checkpoint (must be differentiable).
        *args:         Positional arguments to pass to function.
        use_reentrant: Ignored (for API compatibility only).
        **kwargs:      Keyword arguments to pass to function.

    Returns:
        Output of function(*args, **kwargs), with a custom backward that re-runs
        the forward to recompute activations.
    """
    import lucid
    from lucid._tensor.tensor import Tensor
    from lucid.autograd.function import Function

    class CheckpointFunction(Function):
        @staticmethod
        def forward(ctx: Any, *inputs: Any) -> Any:
            ctx.function = function
            ctx.kwargs = kwargs
            ctx.num_inputs = len(inputs)
            # Save inputs for recomputation (not activations — that's the point)
            ctx.save_for_backward(*inputs)
            with lucid.no_grad():
                return function(*inputs, **kwargs)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
            saved = ctx.saved_tensors
            # Re-run forward with grad enabled to build a fresh computation graph
            inputs_detached = [s.detach().requires_grad_(s.requires_grad) for s in saved]
            with lucid.enable_grad():
                output = ctx.function(*inputs_detached, **ctx.kwargs)

            # Backprop through the recomputed graph
            if not isinstance(output, (list, tuple)):
                output = (output,)
            grads = lucid.autograd.grad(
                list(output),
                [inp for inp in inputs_detached if inp.requires_grad],
                grad_outputs=list(grad_outputs),
                allow_unused=True,
            )

            # Align grad outputs with inputs (None for non-differentiable inputs)
            grad_iter = iter(grads)
            return tuple(
                next(grad_iter) if inp.requires_grad else None
                for inp in inputs_detached
            )

    return CheckpointFunction.apply(*args)
