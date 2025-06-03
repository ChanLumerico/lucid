from dataclasses import dataclass
from typing import Literal, Self

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _NumPyArray


# __all__ = ["RCNN"]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = lucid.arange(n, dtype=lucid.Int32)
        self.size = lucid.ones(n, dtype=lucid.Int32)
        self.int_diff = lucid.zeros(n)

    def _p(self, idx: int) -> int:
        return self.parent[idx].item()

    def find(self, x: int) -> int:
        root = x
        while self._p(root) != root:
            root = self._p(root)

        while self._p(x) != x:
            nxt = self._p(x)
            self.parent[x] = root
            x = nxt

        return root

    def union(self, x: int, y: int, weight: float) -> int:
        x_root, y_root = self.find(x), self.find(y)
        if x_root == y_root:
            return x_root

        if self.size[x_root].item() < self.size[y_root].item():
            x_root, y_root = y_root, x_root

        self.parent[y_root] = x_root
        self.size[x_root] = self.size[x_root] + self.size[y_root]

        self.int_diff[x_root] = max(
            self.int_diff[x_root].item(), self.int_diff[y_root].item(), weight
        )
        return x_root

    def component_size(self, x: int) -> int:
        return self.size[self.find(x)].item()


def _compute_edges(
    image: Tensor, connectivity: Literal[4, 8] = 8
) -> tuple[Tensor, Tensor, Tensor]:
    H, W = image.shape[:2]
    idx = lucid.arange(H * W, dtype=lucid.Int32).reshape(H, W)

    def _color_dist(a: Tensor, b: Tensor) -> Tensor:
        diff = a.astype(lucid.Float32) - b.astype(lucid.Float32)
        if diff.ndim == 2:
            return lucid.abs(diff)
        return lucid.sqrt(lucid.sum(diff * diff, axis=-1))

    displacements = [(0, 1), (1, 0)]
    if connectivity == 8:
        displacements += [(1, 1), (1, -1)]

    edges_p, edges_q, edges_w = [], [], []
    for dy, dx in displacements:
        p = idx[max(0, dy) : H - max(0, -dy), max(0, dx) : W - max(0, -dx)].ravel()
        q = idx[max(0, -dy) : H - max(0, dy), max(0, -dx) : W - max(0, dx)].ravel()

        w = _color_dist(
            image[max(0, dy) : H - max(0, -dy), max(0, dx) : W - max(0, -dx)],
            image[max(0, -dy) : H - max(0, dy), max(0, -dx) : W - max(0, dx)],
        ).ravel()

        edges_p.append(p)
        edges_q.append(q)
        edges_w.append(w)

    return (
        lucid.concatenate(edges_p).to(image.device),
        lucid.concatenate(edges_q).to(image.device),
        lucid.concatenate(edges_w).to(image.device),
    )


def _felzenszwalb_segmentation(
    image: Tensor, k: float = 500.0, min_size: int = 20, connectivity: Literal[4, 8] = 8
) -> Tensor:
    C, H, W = image.shape
    img_f32 = image.astype(lucid.Float32)
    img_cl = img_f32[0] if C == 1 else img_f32.transpose((1, 2, 0))

    n_px = H * W
    p, q, w = _compute_edges(img_cl, connectivity)
    order = lucid.argsort(w, kind="mergesort")
    p, q, w = p[order], q[order], w[order]

    p_list, q_list, w_list = p.data.tolist(), q.data.tolist(), w.data.tolist()
    uf = _UnionFind(n_px)

    for i, (pi, qi, wi) in enumerate(zip(p_list, q_list, w_list)):
        if i % 1000 == 0:
            print(f"Merge [{i}/{len(p_list)}]")  #
        Cp, Cq = uf.find(pi), uf.find(qi)
        if Cp == Cq:
            continue

        thresh = min(
            uf.int_diff[Cp].item() + k / uf.component_size(Cp),
            uf.int_diff[Cq].item() + k / uf.component_size(Cq),
        )
        if wi <= thresh:
            uf.union(Cp, Cq, wi)

    for i, (pi, qi, wi) in enumerate(zip(p_list, q_list, w_list)):
        if i % 1000 == 0:
            print(f"Clean [{i}/{len(p_list)}]")  #
        Cp, Cq = uf.find(pi), uf.find(qi)
        if Cp != Cq and (
            uf.component_size(Cp) < min_size or uf.component_size(Cq) < min_size
        ):
            uf.union(Cp, Cq, wi)

    roots = Tensor([uf.find(i) for i in range(n_px)], dtype=lucid.Int32)
    labels = lucid.unique(roots, return_inverse=True)[1]

    return labels.reshape(H, W)


@dataclass
class _Region:
    idx: int
    bbox: tuple[int, int, int, int]
    size: int
    color_hist: _NumPyArray

    def merge(self, other: Self, new_idx: int) -> Self:
        x1 = min(self.bbox[0], other.bbox[0])
        y1 = min(self.bbox[1], other.bbox[1])
        x2 = max(self.bbox[2], other.bbox[2])
        y2 = max(self.bbox[3], other.bbox[3])

        size = self.size + other.size
        color_hist = (
            self.color_hist * self.size + other.color_hist * other.size
        ) / size
        return _Region(new_idx, (x1, y1, x2, y2), size, color_hist)


class _SelectiveSearch(nn.Module):
    def __init__(
        self,
        scales: tuple[float, ...] = (50, 100, 150, 300),
        min_size: int = 20,
        connectivity: Literal[4, 8] = 8,
        max_boxes: int = 2000,
        iou_thresh: float = 0.8,
    ) -> None:
        super().__init__()

    @staticmethod
    def _color_hist(region_pixels: _NumPyArray, bins: int = 8) -> _NumPyArray:
        # TODO: implement `lucid.histogram` series
        ...


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


# TODO: Need to implement `Selective Search` algorithm


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
                m = cls_imgs == img_id
                det_boxes = cls_boxes[m]
                det_scores = cls_scores[m]

                keep = self.nms(det_boxes, det_scores, self.nms_iou_thresh)
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

        return results

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
        return order[keep].astype(lucid.Int)
