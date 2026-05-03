from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dtype import dtype, to_engine_dtype, _ENGINE_TO_DTYPE  # noqa: F401
from lucid._device import device, _device_from_engine  # noqa: F401
from lucid._globals import get_default_dtype, get_default_device

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _unwrap(t: "_C_engine.TensorImpl | Tensor") -> _C_engine.TensorImpl:
    """Return the underlying TensorImpl from a Tensor or TensorImpl."""
    if isinstance(t, _C_engine.TensorImpl):
        return t
    impl = getattr(t, "_impl", None)
    if impl is not None and isinstance(impl, _C_engine.TensorImpl):
        return impl
    raise TypeError(f"Expected Tensor or TensorImpl, got {type(t).__name__}")


def _wrap(impl: _C_engine.TensorImpl) -> "Tensor":
    """Wrap a TensorImpl in a Tensor (zero-copy)."""
    from lucid._tensor.tensor import Tensor
    return Tensor.__new_from_impl__(impl)


def _wrap_or_none(
    impl: "_C_engine.TensorImpl | None",
) -> "Tensor | None":
    """Wrap TensorImpl in Tensor, or return None."""
    return _wrap(impl) if impl is not None else None


def normalize_factory_kwargs(
    dt: "dtype | _C_engine.Dtype | str | None" = None,
    dev: "device | str | None" = None,
    requires_grad: bool = False,
) -> tuple[_C_engine.Dtype, _C_engine.Device, bool]:
    """Resolve dtype/device to engine enums, applying defaults."""
    _dtype = (
        to_engine_dtype(dt) if dt is not None else to_engine_dtype(get_default_dtype())
    )
    if dev is None:
        dev = get_default_device()
    _device = _parse_device(dev)
    return _dtype, _device, requires_grad


def _impl_with_grad(
    impl: _C_engine.TensorImpl, requires_grad: bool
) -> _C_engine.TensorImpl:
    """Return a new TensorImpl with the same data/device/dtype but different requires_grad."""
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(impl.data_as_python()))
    return _C_engine.TensorImpl(arr, impl.device, requires_grad)


def _parse_device(d: "device | _C_engine.Device | str") -> _C_engine.Device:
    """Convert device/string/engine Device → engine Device enum."""
    if isinstance(d, _C_engine.Device):
        return d
    if isinstance(d, device):
        return d._engine
    if isinstance(d, str):
        return device(d)._engine
    raise TypeError(f"Cannot parse device: {d!r}")
