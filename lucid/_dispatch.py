from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dtype import dtype, to_engine_dtype, _ENGINE_TO_DTYPE  # noqa: F401
from lucid._device import device, _device_from_engine  # noqa: F401
from lucid._globals import get_default_dtype, get_default_device

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# 3.2.2: cache the resolved default-dtype/device engine enums so that
# ``normalize_factory_kwargs`` — called once per op invocation (~200–300
# ops per ResNet-18 forward, ~65 k calls per LeNet-5/MNIST epoch) —
# doesn't re-do the lookup + dtype/device parsing on every call.
# Invalidated by ``lucid._globals.set_default_dtype/device``.
_CACHED_DEFAULT_DTYPE_ENUM: _C_engine.Dtype | None = None
_CACHED_DEFAULT_DEVICE_ENUM: _C_engine.Device | None = None


def _default_dtype_enum_cached() -> _C_engine.Dtype:
    global _CACHED_DEFAULT_DTYPE_ENUM
    if _CACHED_DEFAULT_DTYPE_ENUM is None:
        _CACHED_DEFAULT_DTYPE_ENUM = to_engine_dtype(get_default_dtype())
    return _CACHED_DEFAULT_DTYPE_ENUM


def _default_device_enum_cached() -> _C_engine.Device:
    global _CACHED_DEFAULT_DEVICE_ENUM
    if _CACHED_DEFAULT_DEVICE_ENUM is None:
        _CACHED_DEFAULT_DEVICE_ENUM = _parse_device(get_default_device())
    return _CACHED_DEFAULT_DEVICE_ENUM


def _invalidate_default_dtype_cache() -> None:
    """Called by ``lucid._globals.set_default_dtype`` after a change."""
    global _CACHED_DEFAULT_DTYPE_ENUM
    _CACHED_DEFAULT_DTYPE_ENUM = None


def _invalidate_default_device_cache() -> None:
    """Called by ``lucid._globals.set_default_device`` after a change.

    Note: ``lucid._factories.converters`` keeps a *separate* device cache
    for the ndarray-fast-path hot loop; ``set_default_device`` invalidates
    both via the hook installed in ``lucid._globals``.
    """
    global _CACHED_DEFAULT_DEVICE_ENUM
    _CACHED_DEFAULT_DEVICE_ENUM = None


def _unwrap(t: _C_engine.TensorImpl | Tensor) -> _C_engine.TensorImpl:
    """Return the underlying TensorImpl from a Tensor or TensorImpl."""
    if isinstance(t, _C_engine.TensorImpl):
        return t
    # ``getattr(..., default)`` is typed ``Any``; annotate ``object`` so the
    # isinstance guard below narrows reliably across mypy versions (1.19 left
    # the post-isinstance ``Any`` un-narrowed → no-any-return).
    impl: object = getattr(t, "_impl", None)
    if impl is not None and isinstance(impl, _C_engine.TensorImpl):
        return impl
    raise TypeError(f"Expected Tensor or TensorImpl, got {type(t).__name__}")


def _wrap(impl: _C_engine.TensorImpl) -> Tensor:
    """Wrap a TensorImpl in a Tensor (zero-copy)."""
    from lucid._tensor.tensor import Tensor

    return Tensor.__new_from_impl__(impl)


def _wrap_or_none(
    impl: _C_engine.TensorImpl | None,
) -> Tensor | None:
    """Wrap TensorImpl in Tensor, or return None."""
    return _wrap(impl) if impl is not None else None


def normalize_factory_kwargs(
    dt: dtype | type[dtype] | _C_engine.Dtype | str | None = None,
    dev: device | _C_engine.Device | str | None = None,
    requires_grad: bool = False,
) -> tuple[_C_engine.Dtype, _C_engine.Device, bool]:
    """Resolve user-facing ``dtype`` / ``device`` kwargs to engine enums.

    Single dispatch hot path used by every Python factory function
    (``tensor``, ``zeros``, ``randn``, …) to translate the variety of
    accepted argument types (string / class / instance / ``None``) into
    the engine's two enum values plus the ``requires_grad`` flag.
    ``None`` arguments fall through to the cached defaults set by
    :func:`set_default_dtype` / :func:`set_default_device`.

    Parameters
    ----------
    dt : dtype, dtype-class, engine.Dtype, str, or None, optional
        Requested dtype.  ``None`` (default) → current default dtype.
    dev : device, engine.Device, str, or None, optional
        Requested device.  ``None`` (default) → current default device.
    requires_grad : bool, optional
        Pass-through; included in the return tuple unchanged.  Default
        ``False``.

    Returns
    -------
    tuple
        ``(engine_dtype, engine_device, requires_grad)``.  The first two
        are C++ enum values consumable by ``_C_engine`` factories.
    """
    if dt is None:
        _dtype = _default_dtype_enum_cached()
    else:
        _dtype = to_engine_dtype(dt)
    if dev is None:
        _device = _default_device_enum_cached()
    else:
        _device = _parse_device(dev)
    return _dtype, _device, requires_grad


def _impl_with_grad(
    impl: _C_engine.TensorImpl, requires_grad: bool
) -> _C_engine.TensorImpl:
    """Return a new TensorImpl sharing the same storage but with a different requires_grad.
    Uses clone_with_grad — no data copy, no numpy round-trip."""
    return impl.clone_with_grad(requires_grad)


def _parse_device(d: device | _C_engine.Device | str) -> _C_engine.Device:
    """Convert device/string/engine Device → engine Device enum."""
    if isinstance(d, _C_engine.Device):
        return d
    if isinstance(d, device):
        return d._engine
    if isinstance(d, str):
        return device(d)._engine
    raise TypeError(f"Cannot parse device: {d!r}")
