from typing import Callable, List, Any

import lucid
import lucid.nn as nn

from lucid._tensor import Tensor
from lucid.types import _BuiltinNumeric, Numeric


class Compose:
    def __init__(self, transforms: List[Callable[[Any], Any]]) -> None:
        for t in transforms:
            if not isinstance(t, nn.Module):
                raise TypeError(
                    f"Expected a Module instance, got {type(t).__name__}",
                )
        self.transforms = transforms

    def __call__(self, x: Any) -> Any | Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self) -> str:
        transform_names = ", ".join(
            [type(transform).__name__ for transform in self.transforms]
        )
        return f"{type(self).__name__}([{transform_names}])"


class ToTensor(nn.Module):
    def __init__(
        self,
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: _BuiltinNumeric | Numeric | None = None,
    ) -> None:
        super().__init__()
        self.requires_grad = requires_grad
        self.keep_grad = keep_grad
        self.dtype = dtype

    def forward(self, x: Any) -> Tensor:
        if self.dtype is None:
            if hasattr(x, "dtype"):
                self.dtype = x.dtype
            else:
                self.dtype = None

        return lucid.to_tensor(x, self.requires_grad, self.keep_grad, self.dtype)
