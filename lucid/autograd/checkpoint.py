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
* ``use_reentrant=True`` (the default, for compatibility) derives the
  output's ``requires_grad`` from the positional inputs alone, so a
  segment fed only constants produces no gradient for the parameters it
  closes over — silently, while the same block without ``checkpoint``
  trains.  ``use_reentrant=False`` keeps the node in the graph and is the
  safer choice; it is the default in the reference framework for the
  same reason.
"""

from typing import Callable, TYPE_CHECKING, cast, final, override

from lucid._factories.creation import zeros
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
        ``True`` (default, kept for backward compatibility) ties the
        recomputation to the positional inputs: if none of them requires
        grad, backward never runs and nothing inside the segment receives
        a gradient — including parameters it closed over.  ``False``
        anchors the node in the graph so the segment is always recomputed
        and those parameters accumulate normally.  Prefer ``False``
        unless something depends on the old behaviour.
    **kwargs
        Extra keyword arguments forwarded to *function* on both the
        forward and recomputation passes.

    Returns
    -------
    Tensor
        Output of ``function(*args, **kwargs)``.

    Examples
    --------
    >>> import lucid
    >>> W = lucid.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    >>> b = lucid.tensor([0.0, 0.0], requires_grad=True)
    >>> x = lucid.tensor([[1.0, -2.0]], requires_grad=True)
    >>> def block(x):
    ...     return lucid.nn.functional.relu(x @ W + b)
    >>> y = lucid.autograd.checkpoint(block, x, use_reentrant=False)
    >>> y.sum().backward()
    >>> x.grad
    tensor([[1., 0.]])
    """
    fn = function
    kw = kwargs
    n_args = len(args)
    # Non-reentrant mode appends one extra input (see below), so backward
    # has to know how many gradients the caller is actually owed.
    n_inputs = n_args if use_reentrant else n_args + 1

    @final
    class _CheckpointFn(Function):
        @override
        @staticmethod
        def forward(ctx: FunctionCtx, *inputs: Tensor) -> Tensor:
            real = inputs[:n_args]
            ctx.save_for_backward(*real)
            # Run without tracking so intermediate activations are NOT stored.
            with no_grad():
                output = fn(*real, **kw)
            return output

        @override
        @staticmethod
        def backward(  # type: ignore[override]
            ctx: FunctionCtx, grad_output: Tensor
        ) -> tuple[Tensor | None, ...]:
            inputs = ctx.saved_tensors

            # Detach inputs so the re-run doesn't accumulate into their .grad
            # before we collect the fresh gradients below.
            detached = tuple(t.detach().requires_grad_(t.requires_grad) for t in inputs)

            # Re-run the forward segment to rebuild the local graph.  Anything
            # the segment closed over — module parameters, most often — is
            # part of that graph and accumulates through the backward below.
            with enable_grad():
                output = fn(*detached, **kw)

            # Backward through the re-computed graph.  The seed arrives
            # shaped ``(1,)`` for a 0-d output — harmless for an ordinary
            # backward, which broadcasts, but ``Tensor.backward`` checks the
            # shape exactly, so a segment returning a scalar (a loss block, a
            # pooled embedding) could not be checkpointed at all.
            seed = grad_output
            if tuple(seed.shape) != tuple(output.shape):
                seed = seed.reshape(output.shape)
            output.backward(seed)

            grads: tuple[Tensor | None, ...] = tuple(
                t.grad if t.requires_grad else None for t in detached
            )
            # Trailing ``None`` for the anchor, when one was added.
            return grads if n_inputs == n_args else grads + (None,)

    if use_reentrant:
        return cast("Tensor", _CheckpointFn.apply(*args))

    # ── non-reentrant ────────────────────────────────────────────────────
    #
    # The reentrant form derives the output's ``requires_grad`` from its
    # positional inputs alone.  A segment whose inputs are all constants but
    # which closes over trainable parameters — a first layer fed raw data, a
    # block after a frozen stem — therefore produces an output that requires
    # no grad, so backward never runs and *nothing* is recomputed: the
    # parameters silently receive no gradient at all, while the identical
    # block without ``checkpoint`` trains normally.
    #
    # Passing a trainable scalar the segment never reads is enough to keep
    # the node in the graph.  Its own gradient is discarded; what matters is
    # that backward fires, the segment is recomputed under ``enable_grad``,
    # and the closed-over parameters accumulate.
    anchor = zeros(1, requires_grad=True)
    return cast("Tensor", _CheckpointFn.apply(*args, anchor))


__all__ = ["checkpoint"]
