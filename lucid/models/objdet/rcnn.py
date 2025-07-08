import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from .util import SelectiveSearch, apply_deltas, nms


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

        widths = boxes[:, 2] - boxes[:, 0] + 1
        heights = boxes[:, 3] - boxes[:, 1] + 1
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
        self.ss = SelectiveSearch()
        self.warper = _RegionWarper(warper_output_size)
        self.svm = _LinearSVM(feat_dim, num_classes)
        self.bbox_reg = _BBoxRegressor(feat_dim, num_classes)

        self.image_means: nn.Buffer
        self.register_buffer(
            "image_means", lucid.Tensor(image_means).reshape(1, 3, 1, 1) / pixel_scale
        )

        self.nms_iou_thresh = nms_iou_thresh
        self.score_thresh = score_thresh
        self.add_one = 1.0 if add_one else 0.0

    def forward(
        self,
        images: Tensor,
        rois: list[Tensor] | None = None,
        *,
        return_feats: bool = False,
    ) -> tuple[Tensor, ...]:
        images = images / lucid.max(images).clip(min_value=1.0)
        images = images - self.image_means

        if rois is None:
            rois = [self.ss(img) for img in images]
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
        self, images: Tensor, *, max_det_per_img: int = 100
    ) -> list[dict[str, Tensor]]:
        device = images.device
        _, _, H, W = images.shape

        rois = [self.ss(img) for img in images]
        cls_scores, bbox_deltas = self(images, rois=rois)
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
            cls_boxes = apply_deltas(
                boxes_all[keep_mask], bbox_deltas[keep_mask, c], self.add_one
            )
            cls_scores = cls_probs[keep_mask]
            cls_imgs = img_indices[keep_mask]

            for img_id in cls_imgs.unique():
                m = cls_imgs == img_id
                det_boxes = cls_boxes[m]
                det_scores = cls_scores[m]

                keep = nms(det_boxes, det_scores, self.nms_iou_thresh)
                if keep.size == 0:
                    continue

                res = results[int(img_id.item())]
                res["boxes"].append(det_boxes[keep])
                res["scores"].append(det_scores[keep])
                res["labels"].append(
                    lucid.full((keep.size,), c, dtype=int, device=device)
                )

        for res in results:
            if not res["boxes"]:
                res["boxes"] = lucid.empty(0, 4, device=device)
                res["scores"] = lucid.empty(0, device=device)
                res["labels"] = lucid.empty(0, dtype=int, device=device)
            else:
                res["boxes"] = lucid.concatenate(res["boxes"])
                res["scores"] = lucid.concatenate(res["scores"])
                res["labels"] = lucid.concatenate(res["labels"])

                if res["scores"].size > max_det_per_img:
                    topk = lucid.topk(res["scores"], k=max_det_per_img)[1]
                    res["boxes"] = res["boxes"][topk]
                    res["scores"] = res["scores"][topk]
                    res["labels"] = res["labels"][topk]

        for res in results:
            b = res["boxes"]
            b = b.clip(min_value=0)
            bx = b[:, [0, 2]].clip(max_value=W - 1)
            by = b[:, [1, 3]].clip(max_value=H - 1)

            res["boxes"] = lucid.concatenate(
                [bx[:, :1], by[:, :1], bx[:, 1:], by[:, 1:]], axis=1
            )

        return results
