from typing import ClassVar
from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.objdet.util import nms, iou, DetectionDict


__all__ = ["YOLO_V3", "yolo_v3", "yolo_v3_tiny"]


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
            *_convblock(64, 128, 3, s=2, p=1),
            *[_ResidualBlock(128) for _ in range(2)],
        )
        self.layer4 = nn.Sequential(
            *_convblock(128, 256, 3, s=2, p=1),
            *[_ResidualBlock(256) for _ in range(8)],
        )
        self.layer5 = nn.Sequential(
            *_convblock(256, 512, 3, s=2, p=1),
            *[_ResidualBlock(512) for _ in range(8)],
        )
        self.layer6 = nn.Sequential(
            *_convblock(512, 1024, 3, s=2, p=1),
            *[_ResidualBlock(1024) for _ in range(4)],
        )

        self.num_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(
        self, x: Tensor, classification: bool = False
    ) -> Tensor | tuple[Tensor, ...]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        r1 = self.layer4(x)
        r2 = self.layer5(r1)
        r3 = self.layer6(r2)

        if classification:
            out = self.gap(r3)
            return self.fc(out)

        return r1, r2, r3


class _DarkNet_53_Tiny(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.layer_prev = nn.Sequential(
            *_convblock(3, 16, 3, p=1),
            *_convblock(16, 32, 3, s=2, p=1),
            *_convblock(32, 64, 3, s=2, p=1),
        )
        self.layer4 = nn.Sequential(*_convblock(64, 128, 3, s=2, p=1))
        self.layer5 = nn.Sequential(*_convblock(128, 256, 3, s=2, p=1))
        self.layer6 = nn.Sequential(*_convblock(256, 512, 3, s=2, p=1))

        self.num_classes = num_classes
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

    def forward(
        self, x: Tensor, classification: bool = False
    ) -> Tensor | tuple[Tensor, ...]:
        x = self.layer_prev(x)

        r1 = self.layer4(x)
        r2 = self.layer5(r1)
        r3 = self.layer6(r2)

        if classification:
            out = self.gap(r3)
            return self.fc(out)

        return r1, r2, r3


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
        darknet_out_channels_arr: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        if anchors is None:
            anchors = YOLO_V3.default_anchors
        self.anchors = nn.Buffer(anchors, dtype=lucid.Float32)
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.darknet = _DarkNet_53() if darknet is None else darknet
        if darknet is None:
            dark_out_chs = [256, 512, 1024]
        else:
            if darknet_out_channels_arr is None:
                raise ValueError(
                    "If providing a custom darknet, you must also specify "
                    "`darknet_out_channels_arr`."
                )
            dark_out_chs = darknet_out_channels_arr

        self.head1_pre = nn.Sequential(
            *_convblock(dark_out_chs[2], 512, 1),
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
            *_convblock(dark_out_chs[1] + 256, 256, 1),
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
            *_convblock(dark_out_chs[0] + 128, 128, 1),
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

    def forward(self, x: Tensor) -> tuple[Tensor]:
        r1, r2, r3 = self.darknet(x)

        x = self.head1_pre(r3)
        route = x
        out1 = self.head1_post(x)

        x = self.route1(route)
        x = lucid.concatenate([x, r2], axis=1)
        x = self.head2_pre(x)
        route = x
        out2 = self.head2_post(x)

        x = self.route2(route)
        x = lucid.concatenate([x, r1], axis=1)
        x = self.head3_pre(x)
        out3 = self.head3_post(x)

        return out1, out2, out3

    def _loss_per_scale(self, pred: Tensor, target: Tensor, anchors: Tensor) -> Tensor:
        N = pred.shape[0]
        C = self.num_classes
        B = anchors.shape[0]
        H, W = pred.shape[2:]

        pred = pred.reshape(N, B, 5 + C, H, W).transpose((0, 3, 4, 1, 2))
        target = target.reshape(N, H, W, B, 5 + C)

        stride = self.image_size / H
        ahw = anchors / stride

        obj_mask = target[..., 4:5]
        noobj_mask = 1.0 - obj_mask

        pred_xy_logits = pred[..., 0:2]
        pred_wh_logits = pred[..., 2:4]
        obj_logits = pred[..., 4:5]
        cls_logits = pred[..., 5:]

        tgt_xy_off = target[..., 0:2]
        tgt_wh_log = target[..., 2:4]
        tgt_cls = target[..., 5:]

        pred_xy = F.sigmoid(pred_xy_logits)
        pred_wh = lucid.exp(pred_wh_logits) * ahw.reshape(1, 1, 1, B, 2)
        tgt_wh = lucid.exp(tgt_wh_log) * ahw.reshape(1, 1, 1, B, 2)

        gy, gx = lucid.meshgrid(lucid.arange(H), lucid.arange(W), indexing="ij")
        grid = lucid.stack([gx, gy], axis=-1).reshape(1, H, W, 1, 2)

        pred_xy_px = (pred_xy + grid) * stride
        pred_wh_px = pred_wh * stride

        pred_x1y1 = pred_xy_px - pred_wh_px / 2
        pred_x1y2 = pred_xy_px + pred_wh_px / 2
        pred_boxes_px = lucid.concatenate([pred_x1y1, pred_x1y2], axis=-1)

        tgt_xy_px = (tgt_xy_off + grid) * stride
        tgt_wh_px = tgt_wh * stride

        tgt_x1y1 = tgt_xy_px - tgt_wh_px / 2
        tgt_x1y2 = tgt_xy_px + tgt_wh_px / 2
        tgt_boxes_px = lucid.concatenate([tgt_x1y1, tgt_x1y2], axis=-1)

        pred_boxes_flat = pred_boxes_px.reshape(N, -1, 4)
        tgt_boxes_flat = tgt_boxes_px.reshape(N, -1, 4)
        obj_mask_flat = obj_mask.reshape(N, -1, 1)

        max_iou = lucid.zeros((N, pred_boxes_flat.shape[1], 1), dtype=lucid.Float32)
        ignore_thresh = 0.5
        for n in range(N):
            if obj_mask_flat[n].sum() > 0:
                gt_n = tgt_boxes_flat[n][(obj_mask_flat[n][:, 0] > 0.5)]
                if gt_n.shape[0] > 0:
                    ious = iou(pred_boxes_flat[n], gt_n)
                    max_iou[n] = lucid.max(ious, axis=1, keepdims=True)

        max_iou = max_iou.reshape(N, H, W, B, 1)
        noobj_mask = noobj_mask * (max_iou < ignore_thresh).astype(lucid.Float32)

        box_scale = 2.0 - (tgt_wh_px[..., 0] * tgt_wh_px[..., 1]) / (self.image_size**2)

        w = box_scale * obj_mask[..., 0]
        w = lucid.clip(w, 0.0, None)
        sw = lucid.sqrt(w).reshape(N, H, W, B, 1)

        loss_xy = F.mse_loss(
            F.sigmoid(pred_xy_logits) * sw, tgt_xy_off * sw, reduction="sum"
        )
        loss_wh = F.mse_loss(pred_wh_logits * sw, tgt_wh_log * sw, reduction="sum")

        num_obj = lucid.maximum(obj_mask.sum(), 1.0)

        obj_prob = F.sigmoid(obj_logits)
        cls_prob = F.sigmoid(cls_logits)

        loss_obj = F.binary_cross_entropy(
            obj_prob, lucid.ones_like(obj_prob), weight=obj_mask, reduction="sum"
        )
        loss_noobj = F.binary_cross_entropy(
            obj_prob, lucid.zeros_like(obj_prob), weight=noobj_mask, reduction="sum"
        )
        loss_cls = F.binary_cross_entropy(
            cls_prob, tgt_cls, weight=obj_mask, reduction="sum"
        )

        total_loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_cls
        return total_loss / num_obj

    def get_loss(self, x: Tensor, target: tuple[Tensor]) -> Tensor:
        preds = self.forward(x)
        loss = 0.0
        for p, t, mask in zip(preds, target, self.anchor_masks):
            anchors = self.anchors[mask]
            loss += self._loss_per_scale(p, t, anchors)

        return loss / x.shape[0]

    @lucid.no_grad()
    def predict(
        self, x: Tensor, conf_thresh: float = 0.5, iou_thresh: float = 0.5
    ) -> list[list[DetectionDict]]:
        N = x.shape[0]
        C = self.num_classes
        preds = self.forward(x)

        results: list[list[DetectionDict]] = []
        for i in range(N):
            boxes_list = []
            scores_list = []
            for p, mask in zip(preds, self.anchor_masks):
                B = len(mask)
                H, W = p.shape[2:]
                stride = self.image_size / H
                anchor = self.anchors[mask] / stride

                pi = p[i].reshape(B, 5 + C, H, W).transpose((1, 2, 3, 0))

                pred_xy = F.sigmoid(pi[..., 0:2])
                pred_wh = lucid.exp(pi[..., 2:4]) * anchor.reshape(1, 1, B, 2)
                pred_obj = F.sigmoid(pi[..., 4:5])
                pred_cls = F.sigmoid(pi[..., 5:])

                grid_y, grid_x = lucid.meshgrid(
                    lucid.arange(H), lucid.arange(W), indexing="ij"
                )
                grid = lucid.stack([grid_x, grid_y], axis=-1).reshape(H, W, 1, 2)

                box_xy = (pred_xy + grid) * stride
                box_wh = pred_wh * stride

                box_xy1 = box_xy - box_wh / 2
                box_xy2 = box_xy + box_wh / 2

                box_xy1 = lucid.clip(box_xy1, 0.0, self.image_size - 1)
                box_xy2 = lucid.clip(box_xy2, 0.0, self.image_size - 1)

                boxes = lucid.concatenate([box_xy1, box_xy2], axis=-1).reshape(-1, 4)
                scores = (pred_obj * pred_cls).reshape(-1, C)

                boxes_list.append(boxes)
                scores_list.append(scores)

            boxes_img = lucid.concatenate(boxes_list, axis=0)
            scores_img = lucid.concatenate(scores_list, axis=0)

            image_preds: list[DetectionDict] = []
            for cl in range(C):
                cls_scores = scores_img[:, cl]
                mask = cls_scores > conf_thresh
                if not mask.any():
                    continue

                cls_boxes = boxes_img[mask]
                cls_scores = cls_scores[mask]

                keep = nms(cls_boxes, cls_scores, iou_thresh)
                for j in keep:
                    image_preds.append(
                        {
                            "box": cls_boxes[j].tolist(),
                            "score": cls_scores[j].item(),
                            "class_id": cl,
                        }
                    )

            results.append(image_preds)

        return results


@register_model
def yolo_v3(num_classes: int = 80, **kwargs) -> YOLO_V3:
    return YOLO_V3(num_classes=num_classes, image_size=416, **kwargs)


@register_model
def yolo_v3_tiny(num_classes: int = 80, **kwargs) -> YOLO_V3:
    return YOLO_V3(
        num_classes=num_classes,
        image_size=416,
        darknet=_DarkNet_53_Tiny(),
        darknet_out_channels_arr=[128, 256, 512],
        **kwargs,
    )
