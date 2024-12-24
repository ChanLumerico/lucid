import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["Inception"]


class Inception(nn.Module):  # Beta
    def __init__(self):
        super().__init__()


class _IncepV1Module(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_1x1: int,
        reduce_3x3: int,
        out_channels_3x3: int,
        reduce_5x5: int,
        out_channels_5x5: int,
        out_channels_pool: int,
    ) -> None:
        super().__init__()

        # TODO: Continue after `nn.ConvBNReLU2d`.
