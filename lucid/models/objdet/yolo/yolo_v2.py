from typing import ClassVar
from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.objdet.util import nms, DetectionDict


__all__ = ["YOLO_V2", "yolo_v2", "yolo_v2_tiny"]


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


class _DarkNet_19(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes

        conv: list[nn.Module] = []
        conv += [
            *_convblock(3, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *_convblock(32, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        conv += [
            *_convblock(64, 128, 3),
            *_convblock(128, 64, 1),
            *_convblock(64, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        conv += [
            *_convblock(128, 256, 3),
            *_convblock(256, 128, 1),
            *_convblock(128, 256, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        conv += [
            *_convblock(256, 512, 3),
            *_convblock(512, 256, 1),
            *_convblock(256, 512, 3),
            *_convblock(512, 256, 1),
            *_convblock(256, 512, 3),
        ]
        self.conv = nn.Sequential(*conv)

        self.classifier = nn.Sequential(
            *_convblock(512, 1024, 3),
            *_convblock(1024, 1024, 3),
            nn.Conv2d(1024, self.num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.classifier(x)
        return x


class _DarkNet_19_Tiny(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            *_convblock(3, 16, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *_convblock(16, 32, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *_convblock(32, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *_convblock(64, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *_convblock(128, 256, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *_convblock(256, 512, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            *_convblock(512, 1024, 3),
            nn.Conv2d(1024, self.num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.classifier(x)
        return x


_default_anchors = [
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
]


@nn.set_state_dict_pass_attr("darknet_19")
class YOLO_V2(nn.Module):
    default_anchors: ClassVar[list[tuple[float, float]]] = _default_anchors

    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 5,
        anchors: list[tuple[float, float]] | None = None,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        darknet: nn.Module | None = None,
        route_layer: int | None = None,
        image_size: int = 416,
        use_passthrough: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.use_passthrough = use_passthrough

        if anchors is None:
            anchors = YOLO_V2.default_anchors
        self.anchors = nn.Buffer(anchors, dtype=lucid.Float32)

        if darknet is None:
            self.setattr_raw("darknet_19", _DarkNet_19())
            self.darknet = self.darknet_19.conv
            self.route_layer = 28
        else:
            self.darknet = darknet
            if route_layer is None and self.use_passthrough:
                self.route_layer = self._auto_route_index(self.darknet, image_size)
            else:
                self.route_layer = route_layer

        out_c = self._darknet_out_channels(self.darknet)

        if self.use_passthrough:
            route_c = self._layer_out_channels(self.darknet, self.route_layer)
            self.passthrough_conv = nn.Sequential(
                nn.Conv2d(route_c, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1),
            )

            self.head_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.head_conv1 = nn.Sequential(
                nn.Conv2d(out_c, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            )
            self.head_conv2 = nn.Sequential(
                nn.Conv2d(1024 + 256, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            )

        else:
            self.passthrough_conv = nn.Identity()
            self.head_pool = nn.Identity()

            self.head_conv1 = nn.Sequential(
                nn.Conv2d(out_c, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            )
            self.head_conv2 = nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1),
            )

        self.detector = nn.Conv2d(1024, num_anchors * (5 + num_classes), kernel_size=1)

    @staticmethod
    def _darknet_out_channels(darknet: nn.Module) -> int:
        out_c: int | None = None
        for module in darknet.modules():
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
        if out_c is None:
            raise ValueError("No Conv2d layer found in darknet")
        return out_c

    @staticmethod
    def _layer_out_channels(darknet: nn.Module, idx: int) -> int:
        out_c: int | None = None
        for i, module in enumerate(darknet):
            if isinstance(module, nn.Conv2d):
                out_c = module.out_channels
            if i == idx:
                break
        if out_c is None:
            raise ValueError("No Conv2d layer found in darknet")
        return out_c

    @staticmethod
    def _reorg(x: Tensor, stride: int = 2) -> Tensor:
        n, c, h, w = x.shape
        assert h % stride == 0 and w % stride == 0

        x = x.reshape(n, c, h // stride, stride, w // stride, stride)
        x = x.transpose((0, 1, 3, 5, 2, 4))
        return x.reshape(n, c * stride**2, h // stride, w // stride)

    @staticmethod
    def _auto_route_index(darknet: nn.Module, input_hw: int) -> int:
        x = lucid.zeros(1, 3, input_hw, input_hw)
        shapes = []
        for i, layer in enumerate(darknet):
            x = layer(x)
            shapes.append((i, x.shape))

        for i in range(len(shapes) - 1, -1, -1):
            h, w = shapes[i][1][2:]
            if h == 26 and w == 26:
                return i

        return shapes[-1][0]

    def forward(self, x: Tensor) -> Tensor:
        p = None
        for idx, layer in enumerate(self.darknet):
            x = layer(x)
            if self.use_passthrough and idx == self.route_layer:
                p = x

        x = self.head_pool(x)
        x = self.head_conv1(x)

        if self.use_passthrough:
            p = self.passthrough_conv(p)
            p = self._reorg(p)
            x = lucid.concatenate([p, x], axis=1)

        x = self.head_conv2(x)
        x = self.detector(x)

        N, _, H, W = x.shape
        x = x.transpose((0, 2, 3, 1))
        return x.reshape(N, self.num_anchors * (5 + self.num_classes), H, W)

    def get_loss(self, x: Tensor, target: Tensor) -> Tensor:
        N = x.shape[0]
        pred = self.forward(x)

        B, C = self.num_anchors, self.num_classes
        S = pred.shape[2]

        pred = pred.reshape(N, B, 5 + C, S, S).transpose((0, 3, 4, 1, 2))
        target = target.reshape(N, S, S, B, 5 + C)

        obj_mask = target[..., 4:5]
        noobj_mask = 1 - obj_mask

        pred_xy = F.sigmoid(pred[..., 0:2])
        pred_wh = lucid.exp(pred[..., 2:4]) * self.anchors.reshape(1, 1, 1, B, 2)

        pred_obj = F.sigmoid(pred[..., 4:5])
        pred_cls = pred[..., 5:]

        tgt_xy = target[..., 0:2]
        tgt_wh = target[..., 2:4]
        tgt_cls = target[..., 5:]

        loss_xy = F.mse_loss(pred_xy * obj_mask, tgt_xy * obj_mask, reduction="sum")
        loss_wh = F.mse_loss(pred_wh * obj_mask, tgt_wh * obj_mask, reduction="sum")

        loss_obj = F.mse_loss(pred_obj * obj_mask, obj_mask, reduction="sum")
        loss_noobj = F.mse_loss(
            pred_obj * noobj_mask, lucid.zeros_like(pred_obj), reduction="sum"
        )

        cls_loss_vec = F.cross_entropy(
            pred_cls.reshape(-1, C),
            lucid.argmax(tgt_cls, axis=-1).reshape(-1),
            reduction=None,
        )
        loss_cls = (
            cls_loss_vec.reshape(N, S, S, B) * obj_mask.reshape(N, S, S, B)
        ).sum()

        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        )
        return total_loss / N

    @lucid.no_grad()
    def predict(
        self, x: Tensor, conf_thresh: float = 0.5, iou_thresh: float = 0.5
    ) -> list[list[DetectionDict]]:
        N, _, H, W = x.shape
        B, C = self.num_anchors, self.num_classes
        stride = self.img_size / H

        x = self.forward(x)
        x = x.reshape(N, B, 5 + C, H, W).transpose((0, 3, 4, 1, 2))

        pred_xy = F.sigmoid(x[..., 0:2])
        pred_wh = lucid.exp(x[..., 2:4]) * self.anchors.reshape(1, 1, 1, B, 2)
        pred_obj = F.sigmoid(x[..., 4])
        pred_cls = F.softmax(x[..., 5:], axis=-1)

        grid_y, grid_x = lucid.meshgrid(lucid.arange(H), lucid.arange(W), indexing="ij")
        grid = lucid.stack([grid_x, grid_y], axis=-1).reshape(1, H, W, 1, 2)

        box_xy = (pred_xy + grid) * stride
        box_wh = pred_wh * stride
        box_xy1 = box_xy - box_wh / 2
        box_xy2 = box_xy + box_wh / 2
        boxes = lucid.concatenate([box_xy1, box_xy2], axis=-1)

        results = []
        for i in range(N):
            boxes_i = boxes[i].reshape(-1, 4)
            scores_i = pred_obj[i].reshape(-1, 1) * pred_cls[i].reshape(-1, C)

            image_preds: list[DetectionDict] = []
            for cl in range(C):
                cls_scores = scores_i[:, cl]
                mask = cls_scores > conf_thresh
                if not mask.any():
                    continue

                cls_boxes = boxes_i[mask]
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
def yolo_v2(num_classes: int = 20, **kwargs) -> YOLO_V2:
    return YOLO_V2(num_classes=num_classes, num_anchors=5, image_size=416, **kwargs)


@register_model
def yolo_v2_tiny(num_classes: int = 20, **kwargs) -> YOLO_V2:
    darknet_tiny = _DarkNet_19_Tiny()
    model = YOLO_V2(
        num_classes=num_classes,
        num_anchors=5,
        image_size=416,
        darknet=darknet_tiny.conv,
        route_layer=None,
        use_passthrough=False,
        **kwargs,
    )
    model.setattr_raw("darknet_19", darknet_tiny)
    return model
