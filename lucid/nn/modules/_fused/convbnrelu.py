from typing import Literal
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["ConvBNReLU1d", "ConvBNReLU2d", "ConvBNReLU3d"]


_PaddingStr = Literal["same", "valid"]

_Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
_BN = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]


class _ConvBNReLU(nn.Module):
    def __init__(
        self,
        D: int,
        /,
        in_channels: int,
        out_channels: int,
        bn_num_features: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        # TODO: Begin from here on Christmas!
