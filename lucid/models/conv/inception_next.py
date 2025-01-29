import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


class InceptionDWConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Conv2d(
            gc, gc, kernel_size=square_kernel_size, padding="same", groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size), padding="same", groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, kernel_size=(band_kernel_size, 1), padding="same", groups=gc
        )
        self.split_indices = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = x.split(self.split_indices, axis=1)

        # TODO: Implement further from here (refer to `timm/inceptionnext.py`)
