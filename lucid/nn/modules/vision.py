from typing import Tuple

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.nn.functional import _InterpolateType


__all__ = ["Upsample"]


class Upsample(nn.Module):
    def __init__(
        self,
        size: int | Tuple[int, ...] | None = None,
        scale_factor: float | Tuple[float, ...] | None = None,
        mode: _InterpolateType = "nearest",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if size is None and scale_factor is None:
            raise ValueError("Either 'size' or 'scale_factor' must be specified.")

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def _calculate_size(self, input_: Tensor) -> Tuple[int, ...] | int:
        if self.scale_factor is not None:
            if isinstance(self.scale_factor, (int, float)):
                return tuple(
                    input_.shape[i] * self.scale_factor for i in range(2, input_.ndim)
                )

            elif isinstance(self.scale_factor, tuple):
                return tuple(
                    input_.shape[i + 2] * self.scale_factor[i]
                    for i in range(len(self.scale_factor))
                )

        return self.size

    def forward(self, input_: Tensor) -> Tensor:
        size = self._calculate_size(input_) if self.size is None else self.size
        return F.interpolate(
            input_, size=size, mode=self.mode, align_corners=self.align_corners
        )
