import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["RCNN"]


class _RegionWarper(nn.Module):
    def __init__(self, output_size: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, images: Tensor, rois: list[Tensor]) -> Tensor:
        device = images.device
        _, C, H_img, W_img = images.shape

        M = sum(r.shape[0] for r in rois)
        if M == 0:
            return lucid.empty(0, C, *self.output_size, device=device)

        boxes = lucid.concatenate(rois, axis=0).to(device)
        img_ids = lucid.concatenate(
            [lucid.full((len(r),), i, device=device) for i, r in enumerate(rois)]
        )

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        theta = lucid.zeros(M, 2, 3, device=device)
        theta[:, 0, 0] = widths / (W_img - 1)
        theta[:, 1, 1] = heights / (H_img - 1)
        theta[:, 0, 2] = (2 * ctr_x / (W_img - 1)) - 1
        theta[:, 1, 2] = (2 * ctr_y / (H_img - 1)) - 1

        grid = F.affine_grid(theta, size=(M, C, *self.output_size), align_corners=False)
        flat_imgs = images[img_ids]

        return F.grid_sample(flat_imgs, grid, align_corners=False)


class _LinearSVM(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

    def get_loss(self, scores: Tensor, labels: Tensor, margin: float = 1.0) -> Tensor:
        N = scores.shape[0]
        correct = scores[lucid.arange(N).to(scores.device), labels].unsqueeze(axis=1)

        margins = F.relu(scores - correct + margin)
        margins[lucid.arange(N).to(scores.device), labels] = 0.0

        return margins.sum() / N


class _BBoxRegressor(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(feat_dim, num_classes * 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).reshape(x.shape[0], self.num_classes, 4)


class RCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        *,
        image_means: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_scale: float = 1.0,
        warper_output_size: tuple[int, int] = (224, 224),
        nms_iou_thresh: float = 0.3,
        score_thresh: float = 0.0,
        add_one: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.warper = _RegionWarper(warper_output_size)
        self.svm = _LinearSVM(feat_dim, num_classes)
        self.bbox_reg = _BBoxRegressor(feat_dim, num_classes)

        self.image_means: nn.buffer
        self.register_buffer(
            "image_means", lucid.Tensor(image_means).reshape(1, 3, 1, 1) / pixel_scale
        )

        self.nms_iou_thresh = nms_iou_thresh
        self.score_thresh = score_thresh
        self.add_one = 1.0 if add_one else 0.0

    def forward(
        self, images: Tensor, rois: list[Tensor], *, return_feats: bool = False
    ) -> tuple[Tensor, ...]:
        images = images / lucid.max(images).clip(min_value=1.0)
        images = images - self.image_means

        crops = self.warper(images, rois)
        feats = self.backbone(crops)

        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        feats = feats.flatten(axis=1)

        cls_scores = self.svm(feats)
        bbox_deltas = self.bbox_reg(feats)

        if return_feats:
            return cls_scores, bbox_deltas, feats
        return cls_scores, bbox_deltas

    @lucid.no_grad()
    def predict(
        self, images: Tensor, rois: list[Tensor], *, max_det_per_img: int = 100
    ) -> list[dict[str, Tensor]]:
        device = images.device
        cls_scores, bbox_deltas = self(images, rois)
        probs = F.softmax(cls_scores, axis=1)

        boxes_all = lucid.concatenate(rois).to(device)
        img_indices = lucid.concatenate(
            [lucid.full((len(r),), i, device=device) for i, r in enumerate(rois)]
        )

        num_classes = probs.shape[1]
        results = [{"boxes": [], "scores": [], "labels": []} for _ in images]

        for c in range(1, num_classes):
            cls_probs = probs[:, c]
            keep_mask = cls_probs > self.score_thresh
            if keep_mask.sum().item() == 0:
                continue

            keep_mask = keep_mask.astype(bool)
            cls_boxes = self.apply_deltas(
                boxes_all[keep_mask], bbox_deltas[keep_mask, c], self.add_one
            )
            cls_scores = cls_probs[keep_mask]
            cls_imgs = img_indices[keep_mask]

            for img_id in cls_imgs.unique():
                ...

    @staticmethod
    def apply_deltas(boxes: Tensor, deltas: Tensor, add_one: float = 1.0) -> Tensor:
        widths = boxes[:, 2] - boxes[:, 0] + add_one
        heights = boxes[:, 3] - boxes[:, 1] + add_one
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx, dy, dw, dh = deltas.unbind(axis=-1)
        pred_ctr_x = dx * widths + ctr_x
        pred_crt_y = dy * heights + ctr_y
        pred_w = lucid.exp(dw) * widths
        pred_h = lucid.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_crt_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w - add_one
        y2 = pred_crt_y + 0.5 * pred_h - add_one

        return lucid.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def nms(boxes: Tensor, scores: Tensor, iou_thresh: float = 0.3) -> Tensor:
        N = boxes.shape[0]
        if N == 0:
            return lucid.empty(0, device=boxes.device).astype(lucid.Int)

        _, order = scores.sort(descending=True)
        boxes = boxes[order]

        x1, y1, x2, y2 = boxes.unbind(axis=1)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        xx1 = x1.unsqueeze(axis=1).clip(min_value=x1)
        yy1 = y1.unsqueeze(axis=1).clip(min_value=y1)
        xx2 = x2.unsqueeze(axis=1).clip(max_value=x2)
        yy2 = y2.unsqueeze(axis=1).clip(max_value=y2)

        w = (xx2 - xx1 + 1).clip(min_value=0)
        h = (yy2 - yy1 + 1).clip(min_value=0)

        inter: Tensor = w * h
        iou = inter / (areas.unsqueeze(axis=1) + areas - inter)

        keep_mask: Tensor = lucid.ones(N, dtype=bool, device=iou.device)
        for i in range(N):
            if not keep_mask[i]:
                continue

            keep_mask &= (iou[i] <= iou_thresh) | lucid.eye(
                N, dtype=bool, device=iou.device
            )[i]

        keep = lucid.nonzero(keep_mask).flatten()
        return order[keep]
