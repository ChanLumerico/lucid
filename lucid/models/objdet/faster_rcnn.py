import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from .fast_rcnn import _SlowROIPool
from ._util import apply_deltas, nms, clip_boxes


__all__ = ["FasterRCNN"]


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


class _RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        anchor_generator: _AnchorGenerator,
        pre_nms_top_n: int = 6000,
        post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = _RPNHead(in_channels, anchor_generator.num_anchors)
        self.anchor_generator = anchor_generator
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh

    def forward(self, feature: Tensor, image_shape: tuple[int, int]) -> list[Tensor]:
        B, _, H, W = feature.shape
        logits, deltas = self.head(feature)
        logits = logits.transpose((0, 2, 3, 1)).reshape(B, -1)
        deltas = deltas.transpose((0, 2, 3, 1)).reshape(B, -1, 4)

        anchors = self.anchor_generator.grid_anchors(H, W, feature.device)
        proposals: list[Tensor] = []

        for b in range(B):
            scores = F.sigmoid(logits[b])
            boxes = apply_deltas(anchors, deltas[b], add_one=1.0)
            boxes = clip_boxes(boxes, image_shape)

            keep = scores > self.score_thresh
            if lucid.sum(keep) == 0:
                proposals.append(lucid.empty(0, 4, device=feature.device))
                continue

            scores = scores[keep]
            boxes = boxes[keep]
            order = lucid.argsort(scores, descending=True)
            if self.pre_nms_top_n:
                order = order[: self.pre_nms_top_n]

            boxes = boxes[order]
            scores = scores[order]

            keep_idx = nms(boxes, scores, self.nms_thresh)
            if self.post_nms_top_n:
                keep_idx = keep_idx[: self.post_nms_top_n]

            proposals.append(boxes[keep_idx])

        return proposals


class FasterRCNN(nn.Module):
    NotImplemented
