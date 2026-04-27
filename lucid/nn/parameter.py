"""
lucid.nn.parameter — Parameter and Buffer.

`Parameter` is a `Tensor` flagged as `requires_grad=True` and tracked
by an enclosing `Module`'s `_parameters` registry. `Buffer` is a
non-grad `Tensor` similarly tracked under `_buffers` (used for things
like running mean/variance in BatchNorm).
"""

from __future__ import annotations

from lucid._tensor import Tensor
from lucid.types import Float32, Numeric, _ArrayOrScalar, _DeviceType


__all__ = ["Parameter", "Buffer"]


class Parameter(Tensor):
    """A Tensor that participates in gradient updates.

    Default behavior:
      * `requires_grad=True`
      * Float64 inputs get downcast to Float32 (matches legacy semantics
        — params are typically F32 to keep memory and matmul throughput
        consistent across CPU/GPU).
    """

    def __init__(
        self,
        data: Tensor | _ArrayOrScalar,
        dtype: type | Numeric | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        orig_dtype: Numeric | None = None
        if isinstance(data, Tensor):
            if isinstance(data.dtype, Numeric):
                orig_dtype = data.dtype
            data = data.numpy()
        if (
            dtype is None
            and isinstance(orig_dtype, Numeric)
            and orig_dtype.base_dtype is float
            and orig_dtype.bits == 64
        ):
            dtype = Float32
        super().__init__(data, requires_grad=True, dtype=dtype, device=device)


class Buffer(Tensor):
    """A non-trainable Tensor tracked by an enclosing Module."""

    def __init__(
        self,
        data: Tensor | _ArrayOrScalar,
        dtype: type | Numeric | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.numpy()
        super().__init__(data, requires_grad=False, dtype=dtype, device=device)
