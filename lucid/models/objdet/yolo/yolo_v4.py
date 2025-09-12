from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from lucid.models.imgclf.cspnet import csp_darknet_53
from lucid.models.objdet.util import DetectionDict


__all__ = ["YOLO_V4"]


DEFAULT_ANCHORS: list[list[tuple[int, int]]] = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]
DEFAULT_STRIDES: list[int] = [8, 16, 32]


def _to_xyxy(xywh: Tensor) -> tuple[Tensor, Tensor]:
    c = xywh[..., :2]
    wh = xywh[..., 2:]
    x1y1 = c - wh * 0.5
    x2y2 = c + wh * 0.5
    return x1y1, x2y2


def _bbox_iou_xywh(pred_xywh: Tensor, tgt_xywh: Tensor, eps: float = 1e-7) -> Tensor:
    p1, p2 = _to_xyxy(pred_xywh)
    t1, t2 = _to_xyxy(tgt_xywh)

    inter_lt = lucid.maximum(p1, t1)
    inter_rb = lucid.minimum(p2, t2)
    inter_wh = (inter_rb - inter_lt).clip(0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area_p = (p2[..., 0] - p1[..., 0]).clip(0) * (p2[..., 1] - p1[..., 1]).clip(0)
    area_t = (t2[..., 0] - t1[..., 0]).clip(0) * (t2[..., 1] - t1[..., 1]).clip(0)
    union = area_p + area_t - inter + eps
    return inter / union


def _bbox_iou_ciou(pred_xywh: Tensor, tgt_xywh: Tensor, eps: float = 1e-7) -> Tensor:
    p1, p2 = _to_xyxy(pred_xywh)
    t1, t2 = _to_xyxy(tgt_xywh)
    iou = _bbox_iou_xywh(pred_xywh, tgt_xywh, eps)

    enc_lt = lucid.minimum(p1, t1)
    enc_rb = lucid.maximum(p2, t2)
    enc_wh = (enc_rb - enc_lt).clip(0)
    c2 = enc_wh[..., 0] ** 2 + enc_wh[..., 1] ** 2 + eps

    rho2 = (pred_xywh[..., 0] - tgt_xywh[..., 0]) ** 2
    rho2 += (pred_xywh[..., 1] - tgt_xywh[..., 1]) ** 2

    v = (4 / lucid.pi**2) * (
        lucid.arctan(tgt_xywh[..., 2] / (tgt_xywh[..., 3] + eps))
        - lucid.arctan(pred_xywh[..., 2] / (pred_xywh[..., 3] + eps))
    ) ** 2

    with lucid.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2) - alpha * v


def _iou_xyxy(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)

    tl = lucid.maximum(a[:, None, :2], b[:, :2])
    br = lucid.minimum(a[:, None, 2:], b[:, 2:])
    wh = (br - tl).clip(0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    return inter / (area_a[:, None] + area_b - inter + eps)


def _diou_xyxy(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    iou = _iou_xyxy(a, b)
    ac = (a[:, :2] + a[:, 2:]) * 0.5
    bc = (b[:, :2] + b[:, 2:]) * 0.5

    diff = ac[:, None, :] - bc[None, :, :]
    rho2 = (diff**2).sum(axis=-1)

    enc_tl = lucid.minimum(a[:, None, :2], b[:, :2])
    enc_br = lucid.maximum(a[:, None, 2:], b[:, 2:])
    enc_wh = (enc_br - enc_tl).clip(0)

    c2 = (enc_wh**2).sum(axis=-1) + eps
    return iou - (rho2 / c2)


def _smooth_bce(eps: float = 0.0) -> tuple[float, float]:
    return 1.0 - 0.5 * eps, 0.5 * eps


class _DefaultCSPDarkNet53(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = csp_darknet_53()

    def forward(self, x: Tensor) -> list[Tensor]:
        return self.net.forward_features(x, return_stage_out=True)[-3:]


class _ConvBNAct(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class _SPPBlock(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int) -> None:
        super().__init__()
        self.conv1 = _ConvBNAct(c_in, c_mid, k=1)
        self.conv2 = _ConvBNAct(c_mid, c_mid * 2, k=3)
        self.conv3 = _ConvBNAct(c_mid * 2, c_mid, k=1)

        self.pool = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in (5, 9, 13)]
        )
        self.conv4 = _ConvBNAct(c_mid * 4, c_mid, k=1)
        self.conv5 = _ConvBNAct(c_mid, c_mid * 2, k=3)
        self.conv6 = _ConvBNAct(c_mid * 2, c_out, k=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        pooled = [x] + [pool(x) for pool in self.pool]
        x = lucid.concatenate(pooled, axis=1)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class _PANetNeck(nn.Module):
    def __init__(
        self,
        in_channels: list[int] = (256, 512, 1024),
        out_channels: list[int] = (256, 512, 1024),
    ) -> None:
        super().__init__()
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels

        self.spp = _SPPBlock(c5, c5 // 2, o5)

        self.reduce_c5 = _ConvBNAct(c5, c4, k=1)
        self.c4_lat = _ConvBNAct(c4, c4, k=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c4_td = nn.Sequential(
            _ConvBNAct(c4 * 2, c4, k=1),
            _ConvBNAct(c4, c4, k=3),
            _ConvBNAct(c4, c4, k=1),
            _ConvBNAct(c4, c4, k=3),
            _ConvBNAct(c4, c4, k=1),
        )

        self.reduce_c4 = _ConvBNAct(c4, c3, k=1)
        self.c3_lat = _ConvBNAct(c3, c3, k=1)
        self.c3_out = nn.Sequential(
            _ConvBNAct(c3 * 2, c3, k=1),
            _ConvBNAct(c3, c3, k=3),
            _ConvBNAct(c3, c3, k=1),
            _ConvBNAct(c3, c3, k=3),
            _ConvBNAct(c3, o3, k=1),
        )

        self.down_c3 = _ConvBNAct(o3, c4, k=3, s=2)
        self.c4_out = nn.Sequential(
            _ConvBNAct(c4 * 2, c4, k=1),
            _ConvBNAct(c4, c4, k=3),
            _ConvBNAct(c4, c4, k=1),
            _ConvBNAct(c4, c4, k=3),
            _ConvBNAct(c4, o4, k=1),
        )

        self.down_c4 = _ConvBNAct(o4, c5, k=3, s=2)
        self.c5_out = nn.Sequential(
            _ConvBNAct(c5 * 2, c5, k=1),
            _ConvBNAct(c5, c5, k=3),
            _ConvBNAct(c5, c5, k=1),
            _ConvBNAct(c5, c5, k=3),
            _ConvBNAct(c5, o5, k=1),
        )

    def forward(self, feats: tuple[Tensor]) -> tuple[Tensor]:
        c3, c4, c5 = feats

        p5 = self.spp(c5)
        p4_td = self.c4_td(
            lucid.concatenate([self.c4_lat(c4), self.up(self.reduce_c5(p5))], axis=1)
        )
        p3 = self.c3_out(
            lucid.concatenate([self.c3_lat(c3), self.up(self.reduce_c4(p4_td))], axis=1)
        )
        p4 = self.c4_out(lucid.concatenate([self.down_c3(p3), p4_td], axis=1))
        p5 = self.c5_out(lucid.concatenate([self.down_c4(p4), p5], axis=1))
        return p3, p4, p5


class _YOLOHead(nn.Module):
    def __init__(
        self, in_channels: tuple[int, int, int], num_anchors: int, num_classes: int
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.detect = nn.ModuleList()

        out_per_anchor = 6 + num_classes
        for c in in_channels:
            self.detect.append(
                nn.Sequential(
                    _ConvBNAct(c, c * 2, k=3),
                    nn.Conv2d(c * 2, num_anchors * out_per_anchor, kernel_size=1),
                )
            )

    def forward(self, feats: tuple[Tensor]) -> list[Tensor]:
        return [self.detect[i](f) for i, f in enumerate(feats)]


class YOLO_V4(nn.Module):
    def __init__(
        self,
        num_classes: int,
        anchors: list[list[tuple[int, int]]] | None = None,
        strides: list[int] | None = None,
        backbone: nn.Module | None = None,
        in_channels: tuple[int, int, int] = (256, 512, 1024),
        pos_iou_thr: float = 0.25,
        ignore_iou_thr: float = 0.5,
        obj_balance: tuple[float, float, float] = (4.0, 1.0, 0.4),
        cls_label_smoothing: float = 0.0,
        iou_aware_alpha: float = 0.5,
        iou_branch_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors is not None else DEFAULT_ANCHORS
        self.strides = strides if strides is not None else DEFAULT_STRIDES

        self.backbone = backbone if backbone is not None else _DefaultCSPDarkNet53()
        self.neck = _PANetNeck(in_channels, in_channels)
        self.head = _YOLOHead(in_channels, len(self.anchors[0]), num_classes)

        self.pos_iou_thr = pos_iou_thr
        self.ignore_iou_thr = ignore_iou_thr
        self.obj_balance = obj_balance
        self.cls_eps = cls_label_smoothing

        self.iou_aware_alpha = iou_aware_alpha
        self.iou_branch_weight = iou_branch_weight

        self._bce = nn.BCELoss(reduction=None)

    def foward(self, x: Tensor) -> list[Tensor]:
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck((c3, c4, c5))
        outs = self.head((p3, p4, p5))
        return outs

    def _decode_outputs(
        self, preds: list[Tensor], img_size: tuple[int, int]
    ) -> list[Tensor]: ...

    @lucid.no_grad()
    def _diou_nms_per_img(
        self, det: Tensor, conf_thresh: float, iou_thresh: float, max_det: int = 300
    ) -> Tensor: ...

    def _build_targets(
        self,
        targets: list[Tensor],
        feat_shapes: list[tuple[int, int]],
        device: _DeviceType,
        img_size: tuple[int, int],
    ) -> tuple[Tensor, ...]: ...

    def get_loss(self, x: Tensor, targets: list[Tensor]) -> Tensor: ...

    @lucid.no_grad()
    def predict(
        self, x: Tensor, conf_thresh: float = 0.25, iou_thresh: float = 0.5
    ) -> list[list[DetectionDict]]: ...
