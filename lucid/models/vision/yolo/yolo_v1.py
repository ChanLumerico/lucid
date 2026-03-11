from copy import deepcopy
from dataclasses import dataclass, field
from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.utils import iou

__all__ = ["YOLO_V1", "YOLO_V1Config", "yolo_v1", "yolo_v1_tiny"]


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

arch_config_tiny = [
    (16, 3, 1, 1),
    "M",
    (32, 3, 1, 1),
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    "M",
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]


def _parse_conv_spec(spec: object) -> int:
    if isinstance(spec, tuple):
        if len(spec) != 4 or any(not isinstance(v, int) for v in spec):
            raise ValueError(
                "conv_config tuple entries must contain four integer values"
            )
        out_channels, kernel_size, stride, padding = spec
        if out_channels <= 0 or kernel_size <= 0 or stride <= 0 or padding < 0:
            raise ValueError(
                "conv_config tuple entries must use positive output channels, "
                "kernel sizes, strides, and non-negative padding"
            )
        return out_channels

    if isinstance(spec, str):
        if spec != "M":
            raise ValueError("conv_config string entries must be 'M'")
        return -1

    if isinstance(spec, list):
        if len(spec) != 3 or not isinstance(spec[2], int) or spec[2] <= 0:
            raise ValueError(
                "conv_config repeated-block entries must have the form "
                "[conv1, conv2, positive_repeat_count]"
            )
        conv1_out = _parse_conv_spec(spec[0])
        conv2_out = _parse_conv_spec(spec[1])
        if conv1_out <= 0 or conv2_out <= 0:
            raise ValueError(
                "conv_config repeated-block convolution specs must be tuples"
            )
        return conv2_out

    raise TypeError("conv_config entries must be tuples, 'M', or repeated-block lists")


@dataclass
class YOLO_V1Config:
    in_channels: int = 3
    split_size: int = 7
    num_boxes: int = 2
    num_classes: int = 20
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    conv_config: list[object] = field(default_factory=lambda: deepcopy(arch_config))

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.split_size <= 0:
            raise ValueError("split_size must be greater than 0")
        if self.num_boxes <= 0:
            raise ValueError("num_boxes must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.lambda_coord < 0:
            raise ValueError("lambda_coord must be non-negative")
        if self.lambda_noobj < 0:
            raise ValueError("lambda_noobj must be non-negative")
        if not isinstance(self.conv_config, list) or len(self.conv_config) == 0:
            raise ValueError("conv_config must be a non-empty list")

        self.conv_config = deepcopy(self.conv_config)
        final_out_channels = -1
        for spec in self.conv_config:
            out_channels = _parse_conv_spec(spec)
            if out_channels > 0:
                final_out_channels = out_channels

        if final_out_channels != 1024:
            raise ValueError("conv_config must end with a 1024-channel convolution")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class YOLO_V1(nn.Module):
    def __init__(self, config: YOLO_V1Config) -> None:
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.split_size = config.split_size
        self.num_boxes = config.num_boxes
        self.num_classes = config.num_classes
        self.lambda_coord = config.lambda_coord
        self.lambda_noobj = config.lambda_noobj

        self.darknet = self._create_conv_layers(arch=config.conv_config)
        self.fcs = nn.Sequential(
            nn.Linear(1024 * config.split_size**2, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(
                4096,
                config.split_size**2 * (config.num_boxes * 5 + config.num_classes),
            ),
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

        target_xy = target_box[..., :2]
        target_wh = target_box[..., 2:4] / 2
        target_corners = lucid.concatenate(
            [target_xy - target_wh, target_xy + target_wh], axis=-1
        )

        ious = []
        for b in range(B):
            iou_val = (
                iou(
                    pred_corners[..., b, :].reshape(-1, 4),
                    target_corners.reshape(-1, 4),
                )
                .diagonal()
                .reshape(N, S, S, 1)
            )
            ious.append(iou_val)
        ious = lucid.stack(ious, axis=3)

        best_box = lucid.argmax(ious, axis=3, keepdims=True)
        box_range = lucid.arange(B).reshape(1, 1, 1, B)
        best_mask = (box_range == best_box).astype(lucid.Float32).unsqueeze(axis=-1)

        pred_resp = (pred_boxes * best_mask).sum(axis=3)
        pred_xy = pred_resp[..., :2]
        pred_wh = pred_resp[..., 2:4]
        pred_conf = pred_resp[..., 4:5]

        tgt_xy = target_box[..., :2]
        tgt_wh = target_box[..., 2:4]

        loss_xy = F.mse_loss(pred_xy * obj_mask, tgt_xy * obj_mask, reduction="sum")
        pred_wh_sqrt = lucid.sign(pred_wh) * lucid.sqrt(lucid.abs(pred_wh) + 1e-6)
        tgt_wh_sqrt = lucid.sqrt(tgt_wh.clip(min_value=0) + 1e-6)
        loss_wh = F.mse_loss(
            pred_wh_sqrt * obj_mask,
            tgt_wh_sqrt * obj_mask,
            reduction="sum",
        )

        pred_conf_all = pred_boxes[..., 4]
        best_ious = lucid.max(ious, axis=3, keepdims=True)

        loss_obj = F.mse_loss(
            pred_conf * obj_mask.squeeze(axis=-1),
            best_ious * obj_mask.squeeze(axis=-1),
            reduction="sum",
        )
        loss_noobj = F.mse_loss(
            pred_conf_all * noobj_mask.squeeze(axis=-1),
            lucid.zeros_like(pred_conf_all),
            reduction="sum",
        )

        pred_cls = pred[..., B * 5 :]
        target_cls = target[..., B * 5 :]
        loss_cls = F.mse_loss(
            pred_cls * obj_mask, target_cls * obj_mask, reduction="sum"
        )

        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        )
        return total_loss / N


@register_model
def yolo_v1(num_classes: int = 20, **kwargs) -> YOLO_V1:
    config_kwargs = {
        "in_channels": 3,
        "split_size": 7,
        "num_boxes": 2,
        "num_classes": num_classes,
        "lambda_coord": 5.0,
        "lambda_noobj": 0.5,
        "conv_config": deepcopy(arch_config),
    }
    config_kwargs.update(kwargs)
    return YOLO_V1(YOLO_V1Config(**config_kwargs))


@register_model
def yolo_v1_tiny(num_classes: int = 20, **kwargs) -> YOLO_V1:
    config_kwargs = {
        "in_channels": 3,
        "split_size": 7,
        "num_boxes": 2,
        "num_classes": num_classes,
        "lambda_coord": 5.0,
        "lambda_noobj": 0.5,
        "conv_config": deepcopy(arch_config_tiny),
    }
    config_kwargs.update(kwargs)
    return YOLO_V1(YOLO_V1Config(**config_kwargs))
