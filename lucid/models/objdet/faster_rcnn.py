import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from .fast_rcnn import _SlowROIPool
from ._util import apply_deltas, nms, clip_boxes


class _AnchorGenerator(nn.Module):
    def __init__(
        self, sizes: tuple[int, ...], ratios: tuple[float, ...], stride: int
    ) -> None:
        super().__init__()
        self.stride = stride
        base_anchors = []
        for size in sizes:
            area = float(size * size)
            for r in ratios:
                w = math.sqrt(area / r)
                h = w * r
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        self.base_anchors = Tensor(base_anchors, dtype=lucid.Float32)

    @property
    def num_anchors(self) -> int:
        return len(self.base_anchors)

    def grid_anchors(self, feat_h: int, feat_w: int, device: _DeviceType) -> Tensor:
        shift_x = (lucid.arange(feat_w, device=device) + 0.5) * self.stride
        shift_y = (lucid.arange(feat_h, device=device) + 0.5) * self.stride

        y, x = lucid.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = lucid.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)

        anchors = self.base_anchors.to(device).reshape(1.0 - 1, 4)
        anchors += shifts.reshape(-1, 1, 4)

        return anchors.reshape(-1, 4)


class _RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        deltas = self.bbox_pred(x)

        return logits, deltas
