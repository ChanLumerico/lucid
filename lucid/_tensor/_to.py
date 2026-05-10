"""
Tensor.to() and device/dtype shortcut methods.
Injected into Tensor by _inject_to() after class definition.
"""

from typing import Any, TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dtype import dtype as _dtype_cls, to_engine_dtype
from lucid._dtype import float16, float32, float64, int32, int64, bool_
from lucid._device import device as _device_cls
from lucid._dispatch import _parse_device, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_builtin_bool = bool  # save before any local shadowing


def _inject_to(cls: type) -> None:
    """Attach .to(), .metal(), .cpu(), and dtype-cast methods to Tensor."""

    def to(self: Tensor, *args: _DType | _Device | str, **kwargs: object) -> Tensor:
        """
        Move and/or cast tensor.

        Overloads:
          .to(device)                  → device conversion
          .to(dtype)                   → dtype conversion
          .to(device, dtype)           → both
          .to(other_tensor)            → match other's device & dtype
          .to(device=, dtype=, copy=)
        """
        target_device = self._impl.device
        target_dtype = self._impl.dtype
        copy = _builtin_bool(kwargs.get("copy", False))

        for a in args:
            if isinstance(a, cls):
                target_device = a._impl.device
                target_dtype = a._impl.dtype
            elif isinstance(a, _C_engine.Device):
                target_device = a
            elif isinstance(a, _dtype_cls):
                target_dtype = a._engine
            elif isinstance(a, _device_cls):
                target_device = a._engine
            elif isinstance(a, str):
                target_device = _parse_device(a)

        if "device" in kwargs:
            target_device = _parse_device(kwargs["device"])
        if "dtype" in kwargs:
            target_dtype = to_engine_dtype(kwargs["dtype"])

        same = target_device == self._impl.device and target_dtype == self._impl.dtype
        if same and not copy:
            return self

        # Dtype cast via C++ astype op (CPU: static_cast loop, GPU: mlx::core::astype).
        impl = _C_engine.contiguous(self._impl)
        if target_dtype != impl.dtype:
            impl = _C_engine.astype(impl, target_dtype)
        # Device transfer — SharedStorage fast path vs. legacy Python-buffer path.
        #
        # Strategy:
        #   · Already SharedStorage  → zero-copy relabel (0 memcpy, always optimal)
        #   · Large tensor (≥ 64 KB) → promote to SharedStorage (1 memcpy via Metal),
        #                               then zero-copy relabel — avoids Python overhead
        #   · Small tensor (< 64 KB) → legacy data_as_python path; Metal allocation
        #                               overhead would outweigh the savings here
        #
        # The threshold is empirical: Metal MTLBuffer allocation costs ~µs regardless
        # of size, while data_as_python is near-zero cost for CPU tensors.  For GPU
        # tensors (eval + memcpy) both paths pay ~1 memcpy so shared path wins above
        # the threshold.
        _SHARED_THRESHOLD: int = 64 * 1024  # 64 KB
        if target_device != impl.device:
            rg = self._impl.requires_grad
            if impl.is_metal_shared:
                # Already in shared DRAM — zero-copy relabel, no allocation needed.
                impl = _C_engine.transfer_storage(impl, target_device)
            elif impl.nbytes() >= _SHARED_THRESHOLD:
                # Large tensor: pay 1 Metal allocation + 1 memcpy, avoid Python overhead.
                shared = _C_engine.to_shared_storage(impl)
                impl = _C_engine.transfer_storage(shared, target_device)
            else:
                # Small tensor: legacy path is cheaper (no Metal allocation overhead).
                raw = impl.data_as_python()
                impl = _C_engine.TensorImpl(raw, target_device, rg)
            if impl.requires_grad != rg:
                impl = impl.clone_with_grad(rg)
        else:
            impl = impl.clone_with_grad(self._impl.requires_grad)
        return _wrap(impl)

    def metal(self: Tensor) -> Tensor:
        """Move this tensor to Apple Metal GPU."""
        return to(self, _C_engine.Device.GPU)

    def cpu(self: Tensor) -> Tensor:
        """Move this tensor to CPU."""
        return to(self, _C_engine.Device.CPU)

    def float(self: Tensor) -> Tensor:
        """Cast to float32."""
        return to(self, float32)

    def double(self: Tensor) -> Tensor:
        """Cast to float64."""
        return to(self, float64)

    def half(self: Tensor) -> Tensor:
        """Cast to float16."""
        return to(self, float16)

    def int(self: Tensor) -> Tensor:
        """Cast to int32."""
        return to(self, int32)

    def long(self: Tensor) -> Tensor:
        """Cast to int64."""
        return to(self, int64)

    def bool(self: Tensor) -> Tensor:
        """Cast to bool."""
        return to(self, bool_)

    for _name, _fn in [
        ("to", to),
        ("metal", metal),
        ("cpu", cpu),
        ("float", float),
        ("double", double),
        ("half", half),
        ("int", int),
        ("long", long),
        ("bool", bool),
    ]:
        setattr(cls, _name, _fn)
