import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


# NOTE: WIP
class RegionWarper(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, images: Tensor, rois: list[Tensor]) -> Tensor: ...
