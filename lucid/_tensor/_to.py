"""
Tensor.to() and device/dtype shortcut methods.
Injected into Tensor by _inject_to() after class definition.
"""

from typing import TYPE_CHECKING, cast
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

    def to(
        self: Tensor,
        *args: _dtype_cls | _device_cls | _C_engine.Device | str,
        **kwargs: object,
    ) -> Tensor:
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
                t = cast(Tensor, a)
                target_device = t._impl.device
                target_dtype = t._impl.dtype
            elif isinstance(a, _C_engine.Device):
                target_device = a
            elif isinstance(a, _dtype_cls):
                target_dtype = a._engine
            elif isinstance(a, _device_cls):
                target_device = a._engine
            elif isinstance(a, str):
                target_device = _parse_device(a)

        if "device" in kwargs:
            target_device = _parse_device(
                cast(_device_cls | _C_engine.Device | str, kwargs["device"])
            )
        if "dtype" in kwargs:
            target_dtype = to_engine_dtype(
                cast(
                    _dtype_cls | type[_dtype_cls] | _C_engine.Dtype | str | None,
                    kwargs["dtype"],
                )
            )

        same = target_device == self._impl.device and target_dtype == self._impl.dtype
        if same and not copy:
            return self

        # Dtype cast via C++ astype op (CPU: static_cast loop, GPU: mlx::core::astype).
        impl = _C_engine.contiguous(self._impl)
        if target_dtype != impl.dtype:
            impl = _C_engine.astype(impl, target_dtype)
        # Device transfer.
        #
        # Strategy:
        #   · Already SharedStorage → zero-copy relabel via transfer_storage()
        #     (CPU↔GPU, both directions, no data movement)
        #   · All other tensors     → native upload/download path
        #
        # Why NOT route large tensors through SharedStorage for CPU→GPU:
        #   SharedStorage wraps the buffer as an MLX *external* array backed by
        #   MTLResourceStorageModeShared.  GPU compute kernels that read from shared
        #   Metal memory have measurably lower bandwidth than kernels reading from
        #   MTLResourceStorageModePrivate (GPU-private) buffers — ~130 µs extra
        #   latency per op on 10M-element float32 on M-series chips.
        #
        #   upload_cpu_to_gpu() instead calls mlx::core::copy(external_cpu), which
        #   schedules a Metal blit into a GPU-private buffer.  After the first eval,
        #   the array is fully native and subsequent ops pay no external-array penalty.
        #   This path results in the same single memcpy as SharedStorage, but with
        #   GPU-private destination → faster for all subsequent ops.
        #
        #   SharedStorage remains useful when explicitly requested via
        #   lucid.metal.to_shared() / lucid.metal.shared_tensor() for custom Metal
        #   kernel dispatch where the CPU needs to read/write the same buffer.
        if target_device != impl.device:
            rg = self._impl.requires_grad
            if impl.is_metal_shared:
                # Zero-copy: SharedStorage tensor re-labeled as the target device.
                impl = _C_engine.transfer_storage(impl, target_device)
            else:
                # Native engine path: CPU→GPU uses mlx::core::copy() → GPU-private.
                # GPU→CPU forces eval then downloads to CPU-managed memory.
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
