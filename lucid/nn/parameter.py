from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar, _DeviceType, Float32, Numeric


__all__ = ["Parameter", "Buffer"]


class Parameter(Tensor):
    def __init__(
        self,
        data: Tensor | _ArrayOrScalar,
        dtype: type | Numeric | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        orig_dtype: Numeric | bool | None = None
        if isinstance(data, Tensor):
            if isinstance(data.dtype, Numeric):
                orig_dtype = data.dtype
            data = data.data
        if (
            dtype is None
            and isinstance(orig_dtype, Numeric)
            and orig_dtype.base_dtype is float
            and orig_dtype.bits == 64
        ):
            dtype = Float32
        super().__init__(
            data, requires_grad=True, keep_grad=True, dtype=dtype, device=device
        )


class Buffer(Tensor):
    def __init__(
        self,
        data: Tensor | _ArrayOrScalar,
        dtype: type | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(
            data, requires_grad=False, keep_grad=False, dtype=dtype, device=device
        )
