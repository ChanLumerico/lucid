from typing import ClassVar
from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.objdet.util import nms, iou


def _convblock(
    cin: int, cout: int, k: int, s: int = 1, p: int | None = None
) -> list[nn.Module]:
    return [
        nn.Conv2d(
            cin,
            cout,
            kernel_size=k,
            stride=s,
            padding=p if p is not None else "same",
            bias=False,
        ),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.1),
    ]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        half_channels = channels // 2
        self.block = nn.Sequential(
            *_convblock(channels, half_channels, 1),
            *_convblock(half_channels, channels, 3, p=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _DarkNet_53(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(*_convblock(3, 32, 3))
        self.layer2 = nn.Sequential(
            *_convblock(32, 64, 3, s=2, p=1), _ResidualBlock(64)
        )
        self.layer3 = nn.Sequential(
            *_convblock(64, 128, 3, s=1, p=1),
            *[_ResidualBlock(128) for _ in range(2)],
        )
        self.layer4 = nn.Sequential(
            *_convblock(128, 256, s=2, p=1),
            *[_ResidualBlock(256) for _ in range(8)],
        )
        self.layer5 = nn.Sequential(
            *_convblock(256, 512, s=2, p=1),
            *[_ResidualBlock(512) for _ in range(4)],
        )
        self.layer6 = nn.Sequential(
            *_convblock(512, 1024, 3, s=2, p=1),
            *[_ResidualBlock(1024) for _ in range(4)],
        )

        self.num_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x: Tensor, classification: bool = False) -> Tensor | list[Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        r1 = self.layer4(x)
        r2 = self.layer5(r1)
        r3 = self.layer6(r2)

        if classification:
            out = self.gap(r3)
            return self.fc(out)

        return [r1, r2, r3]


_default_anchors = [
    (10, 13),
    (16, 30),
    (33, 23),
    (30, 61),
    (62, 45),
    (59, 119),
    (116, 90),
    (156, 198),
    (373, 326),
]


class YOLO_V3(nn.Module):
    default_anchors: ClassVar[list[tuple[int, int]]] = _default_anchors

    def __init__(
        self,
        num_classes: int,
        anchors: list[tuple[int, int]] | None = None,
        image_size: int = 416,
        darknet: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        if anchors is None:
            anchors = YOLO_V3.default_anchors
        self.anchors = nn.Buffer(anchors, dtype=lucid.Float32)
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.darknet = _DarkNet_53() if darknet is None else darknet

        self.head1_pre = nn.Sequential(
            *_convblock(1024, 512, 1),
            *_convblock(512, 1024, 3, p=1),
            *_convblock(1024, 512, 1),
            *_convblock(512, 1024, 3, p=1),
            *_convblock(1024, 512, 1),
        )
        self.head1_post = nn.Sequential(
            *_convblock(512, 1024, 3, p=1),
            nn.Conv2d(
                1024,
                len(self.anchor_masks[0]) * (5 + self.num_classes),
                kernel_size=1,
            ),
        )
        self.route1 = nn.Sequential(
            *_convblock(512, 256, 1), nn.Upsample(scale_factor=2, mode="nearest")
        )

        self.head2_pre = nn.Sequential(
            *_convblock(768, 256, 1),
            *_convblock(256, 512, 3, p=1),
            *_convblock(512, 256, 1),
            *_convblock(256, 512, 3, p=1),
            *_convblock(512, 256, 1),
        )
        self.head2_post = nn.Sequential(
            *_convblock(256, 512, 3, p=1),
            nn.Conv2d(
                512,
                len(self.anchor_masks[1]) * (5 + self.num_classes),
                kernel_size=1,
            ),
        )
        self.route2 = nn.Sequential(
            *_convblock(256, 128, 1), nn.Upsample(scale_factor=2, mode="nearest")
        )

        self.head3_pre = nn.Sequential(
            *_convblock(384, 128, 1),
            *_convblock(128, 256, 3, p=1),
            *_convblock(256, 128, 1),
            *_convblock(128, 256, 3, p=1),
            *_convblock(256, 128, 1),
        )
        self.head3_post = nn.Sequential(
            *_convblock(128, 256, 3, p=1),
            nn.Conv2d(
                256,
                len(self.anchor_masks[2]) * (5 + self.num_classes),
                kernel_size=1,
            ),
        )

    def forward(self, x: Tensor) -> list[Tensor]: ...

    def _loss_per_scale(
        self, pred: Tensor, target: Tensor, anchors: Tensor
    ) -> Tensor: ...

    def get_loss(self, x: Tensor, target: list[Tensor]) -> Tensor: ...

    @lucid.no_grad()
    def predict(
        self, x: Tensor, conf_thresh: float = 0.5, iou_thresh: float = 0.5
    ) -> list[list[dict[str, Tensor | int]]]: ...
