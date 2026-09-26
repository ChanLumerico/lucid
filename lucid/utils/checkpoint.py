"""
Gradient checkpointing: trade memory for recomputation during backward.
"""

from typing import Callable
from lucid._tensor.tensor import Tensor


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
        The forward computation to checkpoint.  Randomness it consumes
        (dropout) is replayed on the recompute: the random state is saved
        before the forward and restored for the second run.
    *args : Tensor
        Positional tensor inputs.  Saved by the autograd context and
        passed back to ``function`` on backward.
    use_reentrant : bool, optional
        ``False`` (the default) keeps the segment in the graph even when
        no positional input requires grad, so parameters it closes over
        still train; see :func:`lucid.autograd.checkpoint`.
    **kwargs : object
        Non-tensor keyword arguments forwarded to ``function``.  They
        are *not* differentiated through.

    Returns
    -------
    Tensor or tuple[Tensor, ...]
        Whatever ``function(*args, **kwargs)`` returns — the gradient
        graph routes through :func:`lucid.autograd.checkpoint`, so
        backward triggers the recompute path.

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

    # One implementation, :func:`lucid.autograd.checkpoint`.  This module
    # kept a second copy whose backward took gradients with respect to the
    # explicit inputs only, so every parameter inside the checkpointed
    # function received none and training through it left those weights
    # frozen — and it recomputed with a fresh random state, so a dropout
    # in the segment was differentiated through a different mask.
    from lucid.autograd.checkpoint import checkpoint as _checkpoint

    # ``kwargs`` belong to ``function``; mypy cannot tell them from the
    # checkpoint's own keyword arguments.
    return _checkpoint(
        function, *args, use_reentrant=use_reentrant, **kwargs  # type: ignore[arg-type]
    )
