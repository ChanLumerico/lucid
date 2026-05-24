"""
lucid.compile._signature — Phase 1.4 CacheKey.

A :class:`CacheKey` is the hashable identity of a compiled executable
under the user-facing :class:`CompiledModule`.  Two calls land on the
same executable iff their keys compare equal:

  * Same input *positional* and *keyword* tensor shapes / dtypes.
  * Same device for every input tensor.
  * Same ``training`` flag on the wrapped model (so a ``train()`` ↔
    ``eval()`` flip forces a recompile — BN / Dropout differ).
  * Same ``dynamic`` flag at compile time (the symbolic-batch knob,
    Phase 1.6).
  * Same parameter dtype (so ``.half()`` / ``.float()`` triggers a
    recompile).

The key intentionally does NOT capture:

  * Parameter *values* — those flow in via the cached executable's
    feeds at call time; values mutate between optimizer steps but the
    graph stays valid.
  * Input data values — only the shape / dtype / device of the input
    matters to the captured graph.

Non-tensor positional / keyword arguments must be hashable themselves;
the key embeds them verbatim so a Python ``int`` / ``bool`` / ``str``
distinguishes signatures (useful for ``forward(self, x, training=True)``
style guard arguments).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.module import Module

__all__ = ["TensorSig", "CacheKey", "signature_of"]


@dataclass(frozen=True)
class TensorSig:
    """One tensor's hashable description.

    Captures only what the graph capture path actually depends on:
    rank, per-axis size, dtype name, device type, and a flag whether
    the tensor required grad at trace time (so a leaf vs non-leaf
    input never collides).
    """

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool

    @classmethod
    def of(cls, t: Tensor, *, dynamic_batch: bool = False) -> TensorSig:
        # ``device.type`` (e.g. ``'metal'`` / ``'cpu'``) is hashable;
        # full device repr would include an index slot we don't need.
        # When ``dynamic_batch`` is set we coerce the leading dim of
        # any ≥1-D tensor to ``-1`` so different batch sizes share
        # the same cache entry (Phase 1.6 symbolic-batch opt-in).
        shape = tuple(t.shape)
        if dynamic_batch and len(shape) >= 1:
            shape = (-1, *shape[1:])
        return cls(
            shape=shape,
            dtype=str(t.dtype),
            device=str(t.device.type),
            requires_grad=bool(t.requires_grad),
        )


@dataclass(frozen=True)
class CacheKey:
    """Hashable identity of a CompiledModule signature.

    Two calls share an executable iff their CacheKeys compare equal.
    The dataclass is :func:`frozen` so it can sit in a :class:`set` /
    :class:`dict`; equality / hash both fall out of the field tuple.
    """

    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...]
    training: bool
    dynamic: bool
    # A coarse fingerprint of parameter dtypes / devices — enough that
    # ``.half()`` / ``.to('cpu')`` invalidate the key without us having
    # to walk every parameter on every call.  Captured as a sorted
    # tuple of ``(dtype, device)`` pairs (no name, no shape).
    param_fingerprint: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _arg_sig(value: object, *, dynamic_batch: bool = False) -> object:
    # Lucid Tensor → TensorSig; everything else passes through if
    # hashable.  Iterables (list / tuple) are normalised to a tuple of
    # element sigs so a user-side ``forward(self, [x, y])`` is captured
    # too.  ``dynamic_batch`` propagates into the Tensor case so the
    # leading dim is wildcarded.
    from lucid._tensor.tensor import Tensor

    if isinstance(value, Tensor):
        return TensorSig.of(value, dynamic_batch=dynamic_batch)
    if isinstance(value, (list, tuple)):
        return tuple(_arg_sig(v, dynamic_batch=dynamic_batch) for v in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (str(k), _arg_sig(v, dynamic_batch=dynamic_batch))
                for k, v in value.items()
            )
        )
    # Atoms: int / float / bool / str / None are hashable.  Anything
    # else (e.g. numpy arrays — should not be on this hot path) falls
    # through and will raise on hash if unhashable, which is the
    # signal that this call must take the eager path.
    return value


def _param_fingerprint(model: Module) -> tuple[tuple[str, str], ...]:
    seen: dict[tuple[str, str], int] = {}
    for p in model.parameters():
        key = (str(p.dtype), str(p.device.type))
        seen[key] = seen.get(key, 0) + 1
    return tuple(sorted(seen.keys()))


def signature_of(
    model: Module,
    args: Iterable[object],
    kwargs: dict[str, object],
    *,
    dynamic: bool,
    param_fingerprint: tuple[tuple[str, str], ...] | None = None,
) -> CacheKey:
    """Compute the :class:`CacheKey` for this ``model(*args, **kwargs)`` call.

    ``dynamic=True`` opts into Phase 1.6's symbolic batch axis: the
    leading dim of every input tensor is wildcarded to ``-1`` in the
    captured :class:`TensorSig`, so a call sequence like
    ``compiled(x_bs1); compiled(x_bs32)`` shares a single cache entry.
    ``dynamic`` is itself recorded in :attr:`CacheKey.dynamic` so the
    same model can co-exist as static + dynamic.

    Parameters
    ----------
    model : Module
        Owner of the compiled graph; its :func:`id` is part of the key
        so two different model instances never share a cache slot even
        when their input signatures coincide.
    args : iterable
        Positional call arguments.  Each is summarised via
        :func:`_arg_sig` (tensor → shape/dtype/device, scalar → value).
    kwargs : dict
        Keyword call arguments; summarised the same way as ``args`` and
        sorted by key so insertion order doesn't affect the cache key.
    dynamic : bool
        When ``True`` wildcard the leading axis of every tensor sig
        (dynamic batching).  Recorded in :attr:`CacheKey.dynamic`.
    param_fingerprint : tuple of (dtype_str, device_str) pairs, optional
        Pre-computed parameter fingerprint (dtype × device pairs across
        every parameter, deduplicated).  Defaults to the result of
        :func:`_param_fingerprint(model)`.  Callers that walk parameters
        once per call pass it in to avoid the second walk inside this
        function.

    Returns
    -------
    CacheKey
        Hashable tuple identifying the (model, signature, dynamic)
        triple — used as the dict key in the trace cache.
    """

    arg_sigs = tuple(_arg_sig(a, dynamic_batch=dynamic) for a in args)
    kwarg_sigs = tuple(
        sorted((str(k), _arg_sig(v, dynamic_batch=dynamic)) for k, v in kwargs.items())
    )
    fp = (
        param_fingerprint
        if param_fingerprint is not None
        else _param_fingerprint(model)
    )
    return CacheKey(
        args=arg_sigs,
        kwargs=kwarg_sigs,
        training=bool(getattr(model, "training", False)),
        dynamic=bool(dynamic),
        param_fingerprint=fp,
    )
