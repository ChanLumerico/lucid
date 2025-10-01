import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid import register_model
from lucid.types import _Scalar, _ShapeLike, _DeviceType

import lucid.models.imgclf.efficient as effnet


__all__ = ["EfficientDet"]


class _ConvBlock(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                groups=num_channels,
            ),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels, momentum=0.9997, eps=4e-5),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _BiFPN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.convs = nn.ModuleDict(
            {
                **{f"{i}_up": _ConvBlock(num_channels) for i in range(3, 7)},
                **{f"{i}_down": _ConvBlock(num_channels) for i in range(4, 8)},
            }
        )
        self.ups = nn.ModuleDict(
            {f"{i}": nn.Upsample(scale_factor=2, mode="nearest") for i in range(3, 7)}
        )
        self.downs = nn.ModuleDict(
            {f"{i}": nn.AvgPool2d(kernel_size=2) for i in range(4, 8)}
        )

        self.weights = nn.ModuleDict(
            {
                **{f"{i}_w1": nn.Parameter(lucid.ones(2)) for i in range(3, 7)},
                **{f"{i}_w2": nn.Parameter(lucid.ones(3)) for i in range(4, 8)},
            }
        )
        self.relus = nn.ModuleDict(
            {
                **{f"{i}_w1": nn.ReLU() for i in range(3, 7)},
                **{f"{i}_w2": nn.ReLU() for i in range(4, 8)},
            }
        )

    def _norm_weight(self, weight: Tensor) -> Tensor:
        return weight / (weight.sum(axis=0) + self.eps)

    def _forward_up(self, feats: tuple[Tensor]) -> tuple[Tensor]:
        p3_in, p4_in, p5_in, p6_in, p7_in = feats

        w1_p6_up = self._norm_weight(self.relus["6_w1"](self.weights["6_w1"]))
        p6_up_in = w1_p6_up[0] * p6_in + w1_p6_up[1] * self.ups["6"](p7_in)
        p6_up = self.convs["6_up"](p6_up_in)

        w1_p5_up = self._norm_weight(self.relus["5_w1"](self.weights["5_w1"]))
        p5_up_in = w1_p5_up[0] * p5_in + w1_p5_up[1] * self.ups["5"](p6_up)
        p5_up = self.convs["5_up"](p5_up_in)

        w1_p4_up = self._norm_weight(self.relus["4_w1"](self.weights["4_w1"]))
        p4_up_in = w1_p4_up[0] * p4_in + w1_p4_up[1] * self.ups["4"](p5_up)
        p4_up = self.convs["4_up"](p4_up_in)

        w1_p3_up = self._norm_weight(self.relus["3_w1"](self.weights["3_w1"]))
        p3_up_in = w1_p3_up[0] * p3_in + w1_p3_up[1] * self.ups["3"](p4_up)
        p3_out = self.convs["3_up"](p3_up_in)

        return p3_out, p4_up, p5_up, p6_up

    def _forward_down(
        self, feats: tuple[Tensor], up_feats: tuple[Tensor]
    ) -> tuple[Tensor]:
        _, p4_in, p5_in, p6_in, p7_in = feats
        p3_out, p4_up, p5_up, p6_up = up_feats

        w2_p4_down = self._norm_weight(self.relus["4_w2"](self.weights["4_w2"]))
        p4_down_in = (
            w2_p4_down[0] * p4_in
            + w2_p4_down[1] * p4_up
            + w2_p4_down[2] * self.downs["4"](p3_out)
        )
        p4_out = self.convs["4_down"](p4_down_in)

        w2_p5_down = self._norm_weight(self.relus["5_w2"](self.weights["5_w2"]))
        p5_down_in = (
            w2_p5_down[0] * p5_in
            + w2_p5_down[1] * p5_up
            + w2_p5_down[2] * self.downs["5"](p4_out)
        )
        p5_out = self.convs["5_down"](p5_down_in)

        w2_p6_down = self._norm_weight(self.relus["6_w2"](self.weights["6_w2"]))
        p6_down_in = (
            w2_p6_down[0] * p6_in
            + w2_p6_down[1] * p6_up
            + w2_p6_down[2] * self.downs["6"](p5_out)
        )
        p6_out = self.convs["6_down"](p6_down_in)

        w2_p7_down = self._norm_weight(self.relus["7_w2"](self.weights["7_w2"]))
        p7_down_in = w2_p7_down[0] * p7_in + w2_p7_down[1] * self.downs["7"](p6_out)
        p7_out = self.convs["7_down"](p7_down_in)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        up_feats = self._forward_up(feats)
        down_feats = self._forward_down(feats, up_feats)

        return down_feats


class _BBoxRegresor(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU)

        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.header(x)

        out = x.transpose((0, 2, 3, 1))
        return out.reshape(out.shape[0], -1, 4)


class _Classifier(nn.Module):
    def __init__(
        self, in_channels: int, num_anchors: int, num_classes: int, num_layers: int
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )
        self.act = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.header(x)
        x = self.act(x)

        x = x.transpose((0, 2, 3, 1))
        out = x.reshape(*x.shape[:3], self.num_anchors, self.num_classes)
        return out.reshape(out.shape[0], -1, self.num_classes)


def _generate_anchors(
    base_size: int, ratios: list[_Scalar], scales: list[_Scalar], device: _DeviceType
) -> Tensor:
    num_anchors = len(ratios) * len(scales)
    anchors = lucid.zeros(num_anchors, 4)
    anchors[:, 2:] = base_size * lucid.tile(scales, reps=(2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = lucid.sqrt(areas / lucid.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * lucid.repeat(ratios, len(scales))

    anchors[:, 0::2] -= lucid.tile(anchors[:, 2] * 0.5, reps=(2, 1)).T
    anchors[:, 1::2] -= lucid.tile(anchors[:, 3] * 0.5, reps=(2, 1)).T

    return anchors.to(device)


def _shift_anchors(shape: _ShapeLike, stride: int, anchors: Tensor) -> Tensor:
    shift_x = (lucid.arange(0, shape[1]) + 0.5) * stride
    shift_y = (lucid.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = lucid.meshgrid(shift_x, shift_y, indexing="xy")
    shifts = lucid.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())
    ).T

    A = anchors.shape[0]
    K = shifts.shape[0]

    all_anchors = anchors.reshape(1, A, 4) + shifts.reshape(1, K, 4)
    all_anchors = all_anchors.transpose((1, 0, 2)).reshape(K * A, 4)

    return all_anchors


class _Anchors(nn.Module):
    def __init__(
        self,
        pyramid_levels: list[int] | None = None,
        strides: list[int] | None = None,
        sizes: list[int] | None = None,
        ratios: list[int] | None = None,
        scales: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.pyramid_levels = (
            [3, 4, 5, 6, 7] if pyramid_levels is None else pyramid_levels
        )
        self.strides = (
            [2**x for x in self.pyramid_levels] if strides is None else strides
        )
        self.sizes = (
            [2 ** (x + 2) for x in self.pyramid_levels] if sizes is None else sizes
        )
        self.ratios = [0.5, 1.0, 2.0] if ratios is None else ratios
        self.scales = [2**i for i in (0, 1 / 3, 2 / 3)] if scales is None else scales

    def forward(self, x: Tensor) -> Tensor:
        img_shape = Tensor(x.shape[2:], dtype=lucid.Int16, device=x.device)
        img_shapes: list[Tensor] = [
            (img_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels
        ]
        all_anchors = lucid.zeros(
            len(self.pyramid_levels), 4, dtype=lucid.Float32, device=x.device
        )

        for idx, _ in enumerate(self.pyramid_levels):
            anchors = _generate_anchors(
                self.sizes[idx], self.ratios, self.scales, device=x.device
            )
            shifted_anchors = _shift_anchors(
                img_shapes[idx].tolist(), self.strides[idx], anchors
            )
            all_anchors[idx, :] += shifted_anchors.flatten()

        return all_anchors


class _BBoxTransform(nn.Module):
    def __init__(self, mean: Tensor | None, std: Tensor | None = None) -> None:
        super().__init__()
        self.mean = lucid.zeros(4) if mean is None else mean
        self.std = Tensor([0.1, 0.1, 0.2, 0.2]) if std is None else std

    def forward(self, boxes: Tensor, deltas: Tensor) -> Tensor:
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]

        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        def _inv_norm(d: Tensor, s: Tensor, m: Tensor) -> Tensor:
            return d * s + m

        dx, dy, dw, dh = (
            _inv_norm(deltas[:, :, i], self.std[i], self.mean[i]) for i in range(4)
        )

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = lucid.exp(dw) * widths
        pred_h = lucid.exp(dh) * heights

        pred_box_x1 = pred_ctr_x - 0.5 * pred_w
        pred_box_y1 = pred_ctr_y - 0.5 * pred_h
        pred_box_x2 = pred_ctr_x + 0.5 * pred_w
        pred_box_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = lucid.stack(
            [pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2], axis=2
        )
        return pred_boxes


class _ClipBoxes(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, boxes: Tensor, images: Tensor) -> Tensor:
        boxes[:, :, 0] = lucid.clip(boxes[:, :, 0], min_value=0)
        boxes[:, :, 1] = lucid.clip(boxes[:, :, 1], min_value=0)

        boxes[:, :, 2] = lucid.clip(boxes[:, :, 2], min_value=images.shape[3])
        boxes[:, :, 3] = lucid.clip(boxes[:, :, 3], min_value=images.shape[2])

        return boxes


class EfficientDet(nn.Module):
    def __init__(
        self, num_anchors: int = 9, num_classes: int = 80, compound_coef: int = 0.0
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.compound_coef = compound_coef

        self.num_channels = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]

        self.conv3 = nn.Conv2d(40, self.num_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(80, self.num_channels, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(192, self.num_channels, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(
            192, self.num_channels, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                self.num_channels, self.num_channels, kernel_size=3, stride=2, padding=1
            ),
        )

        self.bifpn = nn.Sequential(
            *[_BiFPN(self.num_channels) for _ in range(min(2 + self.compound_coef, 8))]
        )
        self.regressor = _BBoxRegresor(
            self.num_channels, num_anchors, num_layers=3 + self.compound_coef // 3
        )
        self.classifier = _Classifier(
            self.num_channels,
            num_anchors,
            num_classes,
            num_layers=3 + self.compound_coef // 3,
        )

        self.anchors = _Anchors()
        self.boxtrans = _BBoxTransform()
        self.clipbox = _ClipBoxes()
        self.loss = ...  # TODO: Implement general `nn.FocalLoss`

        self.apply(self.init_weights)

        prior = 0.01
        self._header_weights_post_init(
            self.classifier.header, w_val=0.0, b_val=-math.log((1.0 - prior) / prior)
        )
        self._header_weights_post_init(self.regressor.header, w_val=0.0, b_val=0.0)

        self.backbone = ...  # NOTE: Assign appropriate variant of EfficientNet

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            nn.init.normal(m.weight, 0.0, math.sqrt(2 / n))

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1.0)
            nn.init.constant(m.bias, 0.0)

    @staticmethod
    def _header_weights_post_init(m: nn.Module, w_val: float, b_val: float) -> None:
        assert isinstance(m, nn.Conv2d)
        nn.init.constant(m.weight, w_val)
        nn.init.constant(m.bias, b_val)

    # TODO: Continue from here
    NotImplemented
