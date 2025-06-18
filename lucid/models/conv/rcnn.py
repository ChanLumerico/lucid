from dataclasses import dataclass
from typing import Literal, Self

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["RCNN"]


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

    for pi, qi, wi in zip(p_list, q_list, w_list):
        Cp, Cq = uf.find(pi), uf.find(qi)
        if Cp == Cq:
            continue

        thresh = min(
            uf.int_diff[Cp].item() + k / uf.component_size(Cp),
            uf.int_diff[Cq].item() + k / uf.component_size(Cq),
        )
        if wi <= thresh:
            uf.union(Cp, Cq, wi)

    for pi, qi, wi in zip(p_list, q_list, w_list):
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
    color_hist: Tensor

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
        self.scales = scales
        self.min_size = min_size
        self.connectivity = connectivity
        self.max_boxes = max_boxes
        self.iou_thresh = iou_thresh

    @staticmethod
    def _color_hist(region_pixels: Tensor, bins: int = 8) -> Tensor:
        hist = lucid.histogramdd(
            region_pixels.reshape(-1, 3), bins, range=((0, 256),) * 3
        )[0].flatten()

        hist_sum = hist.sum()
        return hist / hist_sum if hist_sum.item() else hist

    @staticmethod
    def _iou(box_a: Tensor, box_b: Tensor) -> float:
        xa1, ya1, xa2, ya2 = box_a.tolist()
        xb1, yb1, xb2, yb2 = box_b.tolist()

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        area_a = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)
        area_b = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)
        return inter / (area_a + area_b - inter)

    @lucid.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 3:
            raise ValueError("Expecting (C, H, W)")

        _, H, W = image.shape
        rgb = image.transpose((1, 2, 0)).astype(lucid.Int16)
        all_boxes: list[tuple[int, int, int, int]] = []

        for k in self.scales:
            labels = _felzenszwalb_segmentation(
                image,
                k=float(k),
                min_size=self.min_size,
                connectivity=self.connectivity,
            )
            n_regions = lucid.max(labels).item() + 1

            regions: dict[int, _Region] = {}
            for rid in range(n_regions):
                mask = labels == rid
                coords = lucid.nonzero(mask.astype(lucid.Int16))
                size = len(coords)
                if size == 0:
                    continue

                ys, xs = coords[:, 0], coords[:, 1]
                bbox = (
                    int(lucid.min(xs).item()),
                    int(lucid.min(ys).item()),
                    int(lucid.max(xs).item()),
                    int(lucid.max(ys).item()),
                )
                color_hist = self._color_hist(rgb[mask.astype(bool)])
                regions[rid] = _Region(rid, bbox, size, color_hist)

            adj: dict[tuple[int, int], float] = {}
            h_a = labels[:, :-1].ravel()
            h_b = labels[:, 1:].ravel()
            v_a = labels[:-1, :].ravel()
            v_b = labels[1:, :].ravel()

            h_neigh = lucid.stack((h_a, h_b), axis=1)
            v_neigh = lucid.stack((v_a, v_b), axis=1)

            def _sim(r1: _Region, r2: _Region) -> float:
                color_sim = lucid.minimum(r1.color_hist, r2.color_hist).sum().item()
                size_sim = 1.0 - (r1.size + r2.size) / float(H * W)

                x1 = min(r1.bbox[0], r2.bbox[0])
                y1 = min(r1.bbox[1], r2.bbox[1])
                x2 = max(r1.bbox[2], r2.bbox[2])
                y2 = max(r1.bbox[3], r2.bbox[3])

                bbox_size = (x2 - x1 + 1) * (y2 - y1 + 1)
                fill_sim = 1.0 - (bbox_size - r1.size - r2.size) / float(H * W)
                return color_sim + size_sim + fill_sim

            for a, b in lucid.concatenate((h_neigh, v_neigh), axis=0):
                ai, bi = int(a.item()), int(b.item())
                if ai == bi:
                    continue
                key = (ai, bi) if ai < bi else (bi, ai)
                adj[key] = _sim(regions[ai], regions[bi])

            for r in regions.values():
                all_boxes.append(r.bbox)

            next_idx = n_regions
            while adj and len(all_boxes) < self.max_boxes:
                (ra, rb), _ = max(adj.items(), key=lambda item: item[1])
                new_region = regions[ra].merge(regions[rb], next_idx)
                next_idx += 1

                del regions[ra]
                del regions[rb]

                neighbors: set[int] = set()
                for i, j in list(adj.keys()):
                    if i in (ra, rb) or j in (ra, rb):
                        adj.pop((i, j))
                        n = j if i in (ra, rb) else i
                        if n not in (ra, rb):
                            neighbors.add(n)

                regions[new_region.idx] = new_region
                all_boxes.append(new_region.bbox)

                for n in neighbors:
                    if n not in regions:
                        continue
                    key = (
                        (n, new_region.idx)
                        if n < new_region.idx
                        else (new_region.idx, n)
                    )
                    adj[key] = _sim(regions[n], new_region)

        unique_boxes: list[tuple[int, int, int, int]] = []
        for b in all_boxes:
            tb = Tensor(b)
            if all(self._iou(tb, Tensor(ub)) <= self.iou_thresh for ub in unique_boxes):
                unique_boxes.append(b)
            if len(unique_boxes) >= self.max_boxes:
                break

        if unique_boxes:
            return Tensor(unique_boxes, dtype=lucid.Int32)
        return lucid.empty(0, 4, dtype=lucid.Int32)


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
        self.ss = _SelectiveSearch()
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

        for res in results:
            b = res["boxes"]
            b = b.clip(min_value=0)
            bx = b[:, [0, 2]].clip(max_value=W - 1)
            by = b[:, [1, 3]].clip(max_value=H - 1)

            res["boxes"] = lucid.concatenate(
                [bx[:, :1], by[:, :1], bx[:, 1:], by[:, 1:]], axis=1
            )

        return results

    @staticmethod
    def apply_deltas(boxes: Tensor, deltas: Tensor, add_one: float = 1.0) -> Tensor:
        widths = boxes[:, 2] - boxes[:, 0] + add_one
        heights = boxes[:, 3] - boxes[:, 1] + add_one

        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx, dy, dw, dh = deltas.unbind(axis=-1)
        pred_w = (lucid.exp(dw) * widths).clip(min_value=add_one)
        pred_h = (lucid.exp(dh) * heights).clip(min_value=add_one)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w - add_one
        y2 = pred_ctr_y + 0.5 * pred_h - add_one

        x1, x2 = lucid.minimum(x1, x2), lucid.maximum(x1, x2)
        y1, y2 = lucid.minimum(y1, y2), lucid.maximum(y1, y2)

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

        xx1 = lucid.maximum(x1.unsqueeze(1), x1.unsqueeze(0))
        yy1 = lucid.maximum(y1.unsqueeze(1), y1.unsqueeze(0))

        xx2 = lucid.minimum(x2.unsqueeze(1), x2.unsqueeze(0))
        yy2 = lucid.minimum(y2.unsqueeze(1), y2.unsqueeze(0))

        # intersection dims (zeros out negatives)
        w = (xx2 - xx1 + 1).clip(min_value=0)
        h = (yy2 - yy1 + 1).clip(min_value=0)
        inter = w * h

        iou = inter / (areas.unsqueeze(1) + areas - inter)

        keep_mask = lucid.ones(N, dtype=bool, device=boxes.device)
        eye = lucid.eye(N, dtype=bool, device=boxes.device)
        for i in range(N):
            if not keep_mask[i]:
                continue
            keep_mask &= (iou[i] <= iou_thresh) | eye[i]

        keep = lucid.nonzero(keep_mask).flatten()
        return order[keep].astype(lucid.Int)
