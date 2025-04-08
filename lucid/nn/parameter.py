from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar, _DeviceType


__all__ = ["Parameter", "Buffer"]


class Parameter(Tensor):
    def __init__(
        self,
        data: Tensor | _ArrayOrScalar,
        dtype: type | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
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
