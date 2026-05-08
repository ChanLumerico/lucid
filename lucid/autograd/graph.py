"""
lucid.autograd.graph — utilities for inspecting and tweaking the
autograd backward graph.

Currently exposes:

  * ``allow_mutation_on_saved_tensors`` — opt-in escape hatch that
    suppresses the version-mismatch check during backward.  When the
    user knows they are mutating a saved tensor in a way that does not
    corrupt the gradient (rare, expert-only), this lets them proceed
    rather than raising ``VersionMismatch``.

  * ``save_on_cpu`` — context manager that requests saved tensors be
    held on CPU memory instead of the active device, freeing
    accelerator memory at the cost of host transfers during backward.
    Currently a documented stub — full integration with built-in op
    Nodes is a separate engine task.  The context-manager API is
    provided so user code that opts into it does not break; the actual
    memory move is a no-op for now.
"""

from contextlib import contextmanager
from typing import Iterator

from lucid._C import engine as _C_engine


@contextmanager
def allow_mutation_on_saved_tensors() -> Iterator[None]:
    """Disable the autograd version-mismatch check inside the block.

    By default, mutating a tensor in-place after it has been saved for
    backward raises ``VersionMismatch`` — the saved activation no longer
    reflects the value the forward pass observed, so gradients would be
    silently wrong.  Some advanced workflows (custom in-place forward
    paths that are mathematically safe to mutate) need to bypass this
    check.

    Inside this context the engine skips the version check.  Outside,
    the previous setting is restored.  **The user takes full
    responsibility for not corrupting gradients** while the flag is on.

    Example::

        x = lucid.tensor([1.0], requires_grad=True)
        y = x * x
        # y saved x for backward.  Mutating x normally raises here:
        with lucid.autograd.graph.allow_mutation_on_saved_tensors():
            x.add_(1.0)         # ok — no VersionMismatch
            y.backward()        # gradients use the original saved x
    """
    prev = _C_engine.is_mutation_on_saved_allowed()
    _C_engine.set_mutation_on_saved_allowed(True)
    try:
        yield
    finally:
        _C_engine.set_mutation_on_saved_allowed(prev)


@contextmanager
def save_on_cpu(pin_memory: bool = False) -> Iterator[None]:
    """Hint that saved tensors should be staged on CPU during backward.

    Mirrors the reference framework's
    ``torch.autograd.graph.save_on_cpu`` API.  Currently a documented
    stub: the context manager is callable and exits cleanly so user
    code that wraps a forward pass with it does not break, but the
    actual host-staging behaviour for built-in op Nodes is not yet
    wired through the engine.  Custom functions written via
    ``lucid.autograd.Function`` may implement this manually by moving
    their saved tensors via ``.to(device='cpu')`` inside ``forward``
    when this flag is observable.

    Parameters
    ----------
    pin_memory : bool, optional
        Forwarded to the host-staging path when implemented; ignored
        in the current stub.

    Notes
    -----
    Full implementation requires the Node base class to expose a hook
    that ``Engine::backward`` can consult when materialising saved
    activations.  Tracked as an engine follow-up.
    """
    # Currently a no-op — the API is present so callers can adopt it
    # ahead of the underlying engine support.
    _ = pin_memory
    yield


__all__ = ["allow_mutation_on_saved_tensors", "save_on_cpu"]
