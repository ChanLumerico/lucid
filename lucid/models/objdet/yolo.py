from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.objdet.util import iou


__all__ = ["YOLO_V1"]


arch_config = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLO_V1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        split_size: int,
        num_boxes: int,
        num_classes: int,
        lambda_obj: float = 5.0,
        lambda_noobj: float = 5.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

        self.darknet = self._create_conv_layers(arch=arch_config)
        self.fcs = nn.Sequential(
            nn.Linear(1024 * split_size**2, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, split_size**2 * (num_boxes * 5 + num_classes)),
        )

    def _create_conv_layers(self, arch: list) -> nn.Sequential:
        layers = []
        in_channels = self.in_channels
        for cfg in arch:
            if isinstance(cfg, tuple):
                layers.append(
                    _ConvBlock(
                        in_channels,
                        cfg[0],
                        kernel_size=cfg[1],
                        stride=cfg[2],
                        padding=cfg[3],
                    )
                )
                in_channels = cfg[0]

            elif isinstance(cfg, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            elif isinstance(cfg, list):
                conv1, conv2, num_repeats = cfg
                for _ in range(num_repeats):
                    layers.append(
                        _ConvBlock(
                            in_channels,
                            conv1[0],
                            kernel_size=conv1[1],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    )
                    layers.append(
                        _ConvBlock(
                            conv1[0],
                            conv2[0],
                            kernel_size=conv2[1],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    )
                    in_channels = conv2[0]

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.darknet(x)
        x = x.flatten(start_axis=1)
        x = self.fcs(x)
        return x

    def get_loss(self, x: Tensor, target: Tensor) -> Tensor:
        N = x.shape[0]
        S, B, C = self.split_size, self.num_boxes, self.num_classes

        pred = self.forward(x).reshape(N, S, S, B * 5 + C)
        target = target.reshape(N, S, S, B * 5 + C)

        obj_mask = target[..., 4:5]
        noobj_mask = 1.0 - obj_mask

        pred_boxes = pred[..., : B * 5].reshape(N, S, S, B, 5)
        target_box = target[..., :5]

        pred_xy = pred_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:4] / 2
        pred_corners = lucid.concatenate(
            [pred_xy - pred_wh, pred_xy + pred_wh], axis=-1
        )

        target_xy = target_box[..., 2:]
        target_wh = target_box[..., 2:4] / 2
        tatget_corners = lucid.concatenate(
            [target_xy - target_wh, target_xy + target_wh], axis=-1
        )

        ious = []
        for b in range(B):
            iou_val = iou(
                pred_corners[..., b, :].reshape(-1, 4),
                tatget_corners.reshape(-1, 4),
            )
            # TODO: implement `lucid.Tensor.diagonal`
