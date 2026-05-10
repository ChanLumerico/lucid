"""
Gradient checkpointing (``lucid.autograd.checkpoint``).

Gradient checkpointing trades compute for memory: instead of storing
all intermediate activations during the forward pass, only the inputs
to a *segment* are kept.  During the backward pass the segment is
re-executed under ``enable_grad`` to rebuild the local computation graph,
then backpropagated through it.

This is especially useful for large models (transformers, deep ResNets)
where GPU memory is the bottleneck.

Usage
-----
.. code-block:: python

    from lucid.autograd import checkpoint

    def segment(x):
        return model_block(x)

    y = checkpoint(segment, x)    # memory-efficient
    loss = criterion(y, target)
    loss.backward()

Limitations
-----------
* Only single-Tensor outputs are supported.  If *function* returns a
  tuple, use a wrapper that stacks/concatenates outputs into one Tensor
  and splits in the caller.
* ``preserve_rng_state`` is accepted but **not implemented** — RNG state
  is not saved/restored around the recomputation.  Set it to ``False``
  when using stochastic layers (Dropout) inside the checkpoint segment.
* ``use_reentrant=False`` (non-reentrant mode) is not yet supported.
"""

from typing import Callable, TYPE_CHECKING, cast

from lucid.autograd._grad_mode import no_grad, enable_grad
from lucid.autograd.function import Function, FunctionCtx

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def checkpoint(
    function: Callable[..., Tensor],
    *args: Tensor,
    preserve_rng_state: bool = True,
    use_reentrant: bool = True,
    **kwargs: object,
) -> Tensor:
    """Run *function* under gradient checkpointing.

    Executes ``function(*args, **kwargs)`` during the forward pass
    **without** tracking intermediate activations (``no_grad`` context).
    During the backward pass the function is re-executed under
    ``enable_grad`` to reconstruct the local autograd graph, and the
    gradients are computed through that graph.

    Parameters
    ----------
    function : callable
        The differentiable segment to checkpoint.  Must accept tensors as
        positional arguments and return a single :class:`~lucid.Tensor`.
    *args : Tensor
        Positional tensor inputs to *function*.
    preserve_rng_state : bool
        Accepted for API compatibility.  RNG state restoration is not yet
        implemented — set to ``False`` when *function* contains stochastic
        layers.
    use_reentrant : bool
        Accepted for API compatibility.  Only the reentrant (default)
        implementation is provided.
    **kwargs
        Extra keyword arguments forwarded to *function* on both the
        forward and recomputation passes.

    Returns
    -------
    Tensor
        Output of ``function(*args, **kwargs)``.

    Examples
    --------
    >>> def block(x):
    ...     return lucid.nn.functional.relu(x @ W + b)
    >>> y = lucid.autograd.checkpoint(block, x)
    >>> y.sum().backward()
    """
    if not use_reentrant:
        raise NotImplementedError(
            "lucid.autograd.checkpoint: use_reentrant=False is not yet supported."
        )

    fn = function
    kw = kwargs

    class _CheckpointFn(Function):
        @staticmethod
        def forward(ctx: FunctionCtx, *inputs: Tensor) -> Tensor:
            ctx.save_for_backward(*inputs)
            # Run without tracking so intermediate activations are NOT stored.
            with no_grad():
                output = fn(*inputs, **kw)
            return output

        @staticmethod
        def backward(  # type: ignore[override]  # narrower signature than Function/Module base by design
            ctx: FunctionCtx, grad_output: Tensor
        ) -> tuple["Tensor | None", ...]:
            inputs = ctx.saved_tensors

            # Detach inputs so the re-run doesn't accumulate into their .grad
            # before we collect the fresh gradients below.
            detached = tuple(t.detach().requires_grad_(t.requires_grad) for t in inputs)

            # Re-run the forward segment to rebuild the local graph.
            with enable_grad():
                output = fn(*detached, **kw)

            # Backward through the re-computed graph.
            output.backward(grad_output)

            return tuple(t.grad if t.requires_grad else None for t in detached)

    return cast("Tensor", _CheckpointFn.apply(*args))


__all__ = ["checkpoint"]
