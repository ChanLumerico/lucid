"""Allocation-free model instantiation context for the docs/summary
pipeline.

Background
----------
Computing a paper-faithful layer summary tree for a model normally
requires actually constructing the model — the only reliable way to
walk ``named_children()`` and read ``Parameter.shape`` is to run each
sub-module's ``__init__``.  That costs **real memory**: ViT-Huge needs
~2 GB, GPT-2-XL ~6 GB, and anything in the hundreds-of-billions
regime is simply unreachable on a developer laptop or a hosted CI
runner.

This module provides a context manager :func:`shadow_alloc` backed by
the C++ engine's thread-local phantom mode (:func:`_C_engine.phantom_mode` /
:func:`_C_engine.set_phantom_mode`):

  - While enabled on the calling thread, every engine factory and op
    returns a real :class:`_C_engine.TensorImpl` whose storage is a
    ``PhantomStorage`` carrying shape, dtype, device, and autograd
    metadata with no backing buffer and no kernel execution.
  - Leaves every Python-level Module / Parameter / init helper
    untouched.  Their ``__init__`` runs end-to-end, holding phantom
    ``TensorImpl`` instances — no allocation, no MLX array, no zero-init
    kernel.

Constraints
-----------
- This context is **only safe for object construction and shape tracing**.
  Reading a phantom tensor's values (``item``, ``tolist``, ``numpy``,
  ``bool``, ``to_bytes``, ``data_ptr``) or writing a phantom into a real
  tensor raises :exc:`_C_engine.LucidError`.
- Random initialization is skipped — fine for the structural summary,
  but never use a shadow-constructed model for training.
- The block belongs to the thread that opens it (thread-local in the C++
  engine).  Every other thread — including one started inside the block —
  keeps allocating and computing real values.
"""

import contextlib
from collections.abc import Iterator

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor


def is_active() -> bool:
    """Return whether the calling thread is inside a :func:`shadow_alloc` block.

    Used by Lucid's introspection paths (param-count helpers, layer
    summary generators, the docs build's ``shadow_alloc=True`` factory
    invocations) to decide whether tensor storage is real or
    metadata-only.

    Returns
    -------
    bool
        ``True`` inside a :func:`shadow_alloc` block opened by the calling
        thread, ``False`` otherwise — even while another thread has one
        open.
    """
    return _C_engine.phantom_mode()


def is_shadow(t: Tensor) -> bool:
    """Return whether ``t`` is a shadow tensor, carrying metadata but no storage.

    Parameters
    ----------
    t : Tensor
        The tensor to inspect.

    Returns
    -------
    bool
        ``True`` when ``t`` was made inside a :func:`shadow_alloc` block,
        ``False`` for a tensor with real storage.
    """
    return t._impl.is_phantom


@contextlib.contextmanager
def shadow_alloc() -> Iterator[None]:
    r"""Construct models without allocating any tensor storage.

    Use as a context manager around factory invocations whose only
    purpose is to inspect structure (e.g. layer-summary generation
    for the docs site).  Inside the block the engine's tensor-creation
    and op entry points return phantom impls; no real memory is reserved
    even for hundreds-of-billions-parameter models.

    The block applies to the calling thread only.  Other threads,
    including ones started inside it, keep allocating and computing
    real values, and blocks open in two threads at once each last
    until their own exit.  A block nested inside another in the same
    thread restores the outer state on exit.

    Examples
    --------
    >>> from lucid.nn._shadow import shadow_alloc
    >>> from lucid.models.vision.resnet import resnet_152
    >>> with shadow_alloc():
    ...     model = resnet_152()  # constructs without GB of RAM
    >>> # Only structural metadata is meaningful afterwards:
    >>> import math
    >>> sum(math.prod(p.shape) for p in model.parameters())  # counted from shapes
    58143808
    """
    previous = _C_engine.phantom_mode()
    _C_engine.set_phantom_mode(True)
    try:
        yield
    finally:
        _C_engine.set_phantom_mode(previous)
