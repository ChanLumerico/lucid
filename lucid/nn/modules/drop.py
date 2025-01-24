import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
import lucid.nn.functional
import lucid.nn.parameter


__all__ = [
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "DropBlock",
    "DropPath",
]


class _DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"Dropout probability must be between 0 and 1, but got {p}."
            )
        self.p = p


class Dropout(_DropoutNd):
    def forward(self, input_: Tensor) -> Tensor:
        return F.dropout(input_, self.p, self.training)


class Dropout1d(_DropoutNd):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=3)
        return F.dropout1d(input_, self.p, self.training)


class Dropout2d(_DropoutNd):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=4)
        return F.dropout2d(input_, self.p, self.training)


class Dropout3d(_DropoutNd):
    def forward(self, input_: Tensor) -> Tensor:
        lucid._check_input_dim(input_, dim=5)
        return F.dropout3d(input_, self.p, self.training)


class AlphaDropout(_DropoutNd):
    def forward(self, input_: Tensor) -> Tensor:
        return F.alpha_dropout(input_, self.p, self.training)


class DropBlock(nn.Module):
    def __init__(self, block_size: int, p: float = 0.1, eps: float = 1e-7) -> None:
        super().__init__()
        self.block_size = block_size
        self.p = p
        self.eps = eps

    def forward(self, input_: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input_

        return F.drop_block(input_, self.block_size, self.p, self.eps)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, input_: Tensor) -> Tensor:
        return F.drop_path(input_, self.drop_prob, self.scale_by_keep)
