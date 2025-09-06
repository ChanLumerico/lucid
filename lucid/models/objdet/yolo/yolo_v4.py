from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


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


# TODO: Continue from here
class _ConvBNAct(nn.Module): ...


class _SPPBlock(nn.Module): ...


class _PANetNeck(nn.Module): ...


class _YOLOHead(nn.Module): ...


class YOLO_V4(nn.Module): ...
