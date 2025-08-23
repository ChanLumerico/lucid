from typing import ClassVar
from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.objdet.util import nms

__all__ = ["YOLO_V3", "yolo_v3"]


def _convblock(cin: int, cout: int, k: int, s: int = 1, p: int | None = None) -> list[nn.Module]:
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
        half = channels // 2
        self.block = nn.Sequential(
            *_convblock(channels, half, 1),
            *_convblock(half, channels, 3, p=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class _DarkNet_53(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(*_convblock(3, 32, 3))
        self.layer2 = nn.Sequential(*_convblock(32, 64, 3, s=2, p=1), _ResidualBlock(64))
        self.layer3 = nn.Sequential(
            *_convblock(64, 128, 3, s=2, p=1),
            _ResidualBlock(128),
            _ResidualBlock(128),
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        c3 = self.layer4(x)
        c4 = self.layer5(c3)
        c5 = self.layer6(c4)
        return c3, c4, c5


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


@nn.set_state_dict_pass_attr("darknet_53")
class YOLO_V3(nn.Module):
    default_anchors: ClassVar[list[tuple[float, float]]] = _default_anchors

    def __init__(
        self,
        num_classes: int,
        anchors: list[tuple[float, float]] | None = None,
        lambda_coord: float = 1.0,
        lambda_noobj: float = 1.0,
        image_size: int = 416,
        darknet: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.img_size = image_size

        if anchors is None:
            anchors = YOLO_V3.default_anchors
        self.anchors = nn.Buffer(anchors, dtype=lucid.Float32)
        self.anchor_masks = [
            [6, 7, 8],
            [3, 4, 5],
            [0, 1, 2],
        ]

        if darknet is None:
            self.setattr_raw("darknet_53", _DarkNet_53())
            self.darknet = self.darknet_53
        else:
            self.darknet = darknet

        # Head for scale 1 (13x13 for 416 input)
        self.head1_pre = nn.Sequential(
            *_convblock(1024, 512, 1),
            *_convblock(512, 1024, 3, p=1),
            *_convblock(1024, 512, 1),
            *_convblock(512, 1024, 3, p=1),
            *_convblock(1024, 512, 1),
        )
        self.head1_post = nn.Sequential(
            *_convblock(512, 1024, 3, p=1),
            nn.Conv2d(1024, len(self.anchor_masks[0]) * (5 + self.num_classes), kernel_size=1),
        )
        self.route1 = nn.Sequential(
            *_convblock(512, 256, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Head for scale 2 (26x26)
        self.head2_pre = nn.Sequential(
            *_convblock(768, 256, 1),
            *_convblock(256, 512, 3, p=1),
            *_convblock(512, 256, 1),
            *_convblock(256, 512, 3, p=1),
            *_convblock(512, 256, 1),
        )
        self.head2_post = nn.Sequential(
            *_convblock(256, 512, 3, p=1),
            nn.Conv2d(512, len(self.anchor_masks[1]) * (5 + self.num_classes), kernel_size=1),
        )
        self.route2 = nn.Sequential(
            *_convblock(256, 128, 1),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Head for scale 3 (52x52)
        self.head3_pre = nn.Sequential(
            *_convblock(384, 128, 1),
            *_convblock(128, 256, 3, p=1),
            *_convblock(256, 128, 1),
            *_convblock(128, 256, 3, p=1),
            *_convblock(256, 128, 1),
        )
        self.head3_post = nn.Sequential(
            *_convblock(128, 256, 3, p=1),
            nn.Conv2d(256, len(self.anchor_masks[2]) * (5 + self.num_classes), kernel_size=1),
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        c3, c4, c5 = self.darknet(x)

        x = self.head1_pre(c5)
        route = x
        out1 = self.head1_post(x)
        x = self.route1(route)
        x = lucid.concatenate([x, c4], axis=1)

        x = self.head2_pre(x)
        route = x
        out2 = self.head2_post(x)
        x = self.route2(route)
        x = lucid.concatenate([x, c3], axis=1)

        x = self.head3_pre(x)
        out3 = self.head3_post(x)

        return [out1, out2, out3]

    def _loss_per_scale(
        self,
        pred: Tensor,
        target: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        N = pred.shape[0]
        B = anchors.shape[0]
        C = self.num_classes
        H, W = pred.shape[2:]

        pred = pred.reshape(N, B, 5 + C, H, W).transpose((0, 3, 4, 1, 2))
        target = target.reshape(N, H, W, B, 5 + C)

        obj_mask = target[..., 4:5]
        noobj_mask = 1 - obj_mask

        pred_xy = F.sigmoid(pred[..., 0:2])
        pred_wh = lucid.exp(pred[..., 2:4]) * anchors.reshape(1, 1, 1, B, 2)
        pred_obj = F.sigmoid(pred[..., 4:5])
        pred_cls = F.sigmoid(pred[..., 5:])

        tgt_xy = target[..., 0:2]
        tgt_wh = target[..., 2:4]
        tgt_cls = target[..., 5:]

        loss_xy = F.mse_loss(pred_xy * obj_mask, tgt_xy * obj_mask, reduction="sum")
        loss_wh = F.mse_loss(pred_wh * obj_mask, tgt_wh * obj_mask, reduction="sum")

        loss_obj = F.binary_cross_entropy(pred_obj * obj_mask, obj_mask, reduction="sum")
        loss_noobj = F.binary_cross_entropy(
            pred_obj * noobj_mask, lucid.zeros_like(pred_obj), reduction="sum"
        )
        loss_cls = F.binary_cross_entropy(
            pred_cls * obj_mask, tgt_cls * obj_mask, reduction="sum"
        )

        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        )
        return total_loss

    def get_loss(self, x: Tensor, target: list[Tensor]) -> Tensor:
        preds = self.forward(x)
        loss = 0.0
        for p, t, mask in zip(preds, target, self.anchor_masks):
            anchors = self.anchors[mask]
            loss += self._loss_per_scale(p, t, anchors)
        return loss / x.shape[0]

    @lucid.no_grad()
    def predict(
        self, x: Tensor, conf_thresh: float = 0.5, iou_thresh: float = 0.5
    ) -> list[list[dict[str, Tensor | int]]]:
        N = x.shape[0]
        C = self.num_classes
        preds = self.forward(x)

        results: list[list[dict[str, Tensor | int]]] = []
        for i in range(N):
            boxes_list = []
            scores_list = []
            for p, mask in zip(preds, self.anchor_masks):
                B = len(mask)
                H, W = p.shape[2:]
                stride = self.img_size / H
                anchor = self.anchors[mask]

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
                boxes = lucid.concatenate([box_xy1, box_xy2], axis=-1).reshape(-1, 4)
                scores = (pred_obj * pred_cls).reshape(-1, C)

                boxes_list.append(boxes)
                scores_list.append(scores)

            boxes_img = lucid.concatenate(boxes_list, axis=0)
            scores_img = lucid.concatenate(scores_list, axis=0)

            image_preds = []
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
                        {"box": cls_boxes[j], "score": cls_scores[j], "class_id": cl}
                    )

            results.append(image_preds)

        return results


@register_model
def yolo_v3(num_classes: int = 80, **kwargs) -> YOLO_V3:
    return YOLO_V3(num_classes=num_classes, image_size=416, **kwargs)

