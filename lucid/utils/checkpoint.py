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
    """Run ``function`` without saving its intermediates; recompute them on backward.

    Memory-for-compute tradeoff used to fit larger models / longer
    sequences in fixed VRAM.  In a normal forward pass autograd stashes
    every intermediate activation needed by the backward pass — for a
    deep transformer block this is the dominant memory cost.  Wrapping
    the block in :func:`checkpoint` runs forward under :func:`no_grad`,
    saves *only the inputs*, and re-executes the block from scratch
    during backward to recompute the intermediates on demand.

    Net effect: peak memory drops from "all activations" to "inputs +
    one block's intermediates at backward time"; wall-clock grows by
    roughly the cost of one extra forward pass per checkpointed block.

    Parameters
    ----------
    function : callable
        The forward computation to checkpoint.  Must be deterministic
        given ``(args, kwargs)`` — if it consumes randomness (dropout,
        random init), seed it explicitly inside the callable, otherwise
        forward and re-forward will disagree.
    *args : Tensor
        Positional tensor inputs.  Saved by the autograd context and
        passed back to ``function`` on backward.
    use_reentrant : bool, optional
        Accepted for API parity with reference frameworks; currently
        ignored — Lucid always uses the non-reentrant strategy (a
        dedicated :class:`Function` subclass with manual recompute).
        Default ``False``.
    **kwargs : object
        Non-tensor keyword arguments forwarded to ``function``.  They
        are *not* differentiated through.

    Returns
    -------
    Tensor or tuple[Tensor, ...]
        Whatever ``function(*args, **kwargs)`` returns — the gradient
        graph routes through :class:`CheckpointFunction` so backward
        triggers the recompute path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.utils.checkpoint import checkpoint
    >>>
    >>> def block(x):
    ...     return (x @ x.T).relu().sum(dim=-1)
    >>>
    >>> x = lucid.randn(4, 8, requires_grad=True)
    >>> y = checkpoint(block, x)         # forward under no_grad; saves only x
    >>> y.sum().backward()               # re-runs `block(x)` to populate grads

    Notes
    -----
    Best applied to a *sequence* of homogeneous blocks (transformer
    layers, ResNet stages) where the per-block memory saving compounds.
    Checkpointing every layer roughly doubles training time; the usual
    recipe is to checkpoint every other layer or once per stage.

    See Also
    --------
    lucid.autograd.Function : the autograd primitive underneath.
    """

    class CheckpointFunction(Function):
        """Custom autograd Function that runs the wrapped callable without saving intermediates, then re-executes it during backward to recompute the activations on demand."""

        @staticmethod
        def forward(ctx: FunctionCtx, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
            """Apply the layer / parametrization to the input."""
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
            """Compute the gradient for the saved input(s)."""
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
