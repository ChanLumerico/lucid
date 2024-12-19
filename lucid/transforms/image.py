from typing import Any

import lucid
import lucid.nn as nn
from lucid._tensor import Tensor


__all__ = ["Normalize", "Resize"]


class Normalize(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.mean = lucid.tensor(mean)
        self.std = lucid.tensor(std)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class Resize(nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Finish after implementing `nn.F.interpolate()`
        return
