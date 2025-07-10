from dataclasses import dataclass
from typing import Literal, Self

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


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


def felzenszwalb_segmentation(
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


class SelectiveSearch(nn.Module):
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

    @lucid.no_grad()
    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 3:
            raise ValueError("Expecting (C, H, W)")

        _, H, W = image.shape
        rgb = image.transpose((1, 2, 0)).astype(lucid.Int16)
        all_boxes: list[tuple[int, int, int, int]] = []

        for k in self.scales:
            labels = felzenszwalb_segmentation(
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
            if all(iou(tb, Tensor(ub)) <= self.iou_thresh for ub in unique_boxes):
                unique_boxes.append(b)
            if len(unique_boxes) >= self.max_boxes:
                break

        if unique_boxes:
            return Tensor(unique_boxes, dtype=lucid.Int32)
        return lucid.empty(0, 4, dtype=lucid.Int32)


def iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    x1a, y1a, x2a, y2a = boxes_a.unbind(axis=1)
    x1b, y1b, x2b, y2b = boxes_b.unbind(axis=1)

    xx1 = lucid.maximum(x1a.unsqueeze(1), x1b.unsqueeze(0))
    yy1 = lucid.maximum(y1a.unsqueeze(1), y1b.unsqueeze(0))
    xx2 = lucid.minimum(x2a.unsqueeze(1), x2b.unsqueeze(0))
    yy2 = lucid.minimum(y2a.unsqueeze(1), y2b.unsqueeze(0))

    w = (xx2 - xx1 + 1).clip(min_value=0)
    h = (yy2 - yy1 + 1).clip(min_value=0)
    inter = w * h

    area_a = (x2a - x1a + 1) * (y2a - y1a + 1)
    area_b = (x2b - x1b + 1) * (y2b - y1b + 1)

    return inter / (area_a.unsqueeze(1) + area_b - inter)


def bbox_to_delta(src: Tensor, target: Tensor, add_one: float = 1.0) -> Tensor:
    sw = src[:, 2] - src[:, 0] + add_one
    sh = src[:, 3] - src[:, 1] + add_one
    sx = src[:, 0] + 0.5 * sw
    sy = src[:, 1] + 0.5 * sh

    tw = target[:, 2] - target[:, 0] + add_one
    th = target[:, 3] - target[:, 1] + add_one
    tx = target[:, 0] + 0.5 * tw
    ty = target[:, 1] + 0.5 * th

    dx = (tx - sx) / sw
    dy = (ty - sy) / sh
    dw = lucid.log(tw / sw)
    dh = lucid.log(th / sh)

    return lucid.stack([dx, dy, dw, dh], axis=1)


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


def clip_boxes(boxes: Tensor, image_shape: tuple[int, int]) -> Tensor:
    H, W = image_shape
    x1 = lucid.clip(boxes[:, 0], 0, W - 1)
    y1 = lucid.clip(boxes[:, 1], 0, H - 1)
    x2 = lucid.clip(boxes[:, 2], 0, W - 1)
    y2 = lucid.clip(boxes[:, 3], 0, H - 1)

    return lucid.stack([x1, y1, x2, y2], axis=-1)


class ROIAlign(nn.Module):
    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, images: Tensor, rois: Tensor, roi_idx: Tensor) -> Tensor:
        C = images.shape[1]
        ph, pw = self.output_size
        device = images.device

        x1, y1, x2, y2 = rois.unbind(axis=1)
        valid = (x2 > x1) & (y2 > y1)
        if lucid.sum(valid) == 0:
            return lucid.empty(0, C, ph, pw, device=device)

        idx = valid.nonzero().squeeze(axis=1)
        rois = rois[idx]
        roi_idx = roi_idx[idx]
        x1, y1, x2, y2 = rois.unbind(axis=1)

        if pw > 1:
            xs = lucid.arange(pw, dtype=lucid.Float32, device=device) / (pw - 1)
        else:
            xs = lucid.full((1,), 0.5, dtype=lucid.Float32, device=device)

        if ph > 1:
            ys = lucid.arange(ph, dtype=lucid.Float32, device=device) / (ph - 1)
        else:
            ys = lucid.full((1,), 0.5, dtype=lucid.Float32, device=device)

        grid_y_base, grid_x_base = lucid.meshgrid(ys, xs, indexing="ij")
        grid_x_base = grid_x_base.unsqueeze(axis=0)
        grid_y_base = grid_y_base.unsqueeze(axis=0)

        x1_ = x1.unsqueeze(axis=-1).unsqueeze(axis=-1)
        y1_ = y1.unsqueeze(axis=-1).unsqueeze(axis=-1)
        w_ = (x2 - x1).unsqueeze(axis=-1).unsqueeze(axis=-1)
        h_ = (y2 - y1).unsqueeze(axis=-1).unsqueeze(axis=-1)

        grid_x = x1_ + w_ * grid_x_base
        grid_y = y1_ + h_ * grid_y_base
        grid = lucid.stack([grid_y * 2 - 1, grid_x * 2 - 1], axis=-1)

        feat_per_roi = images[roi_idx]
        out = F.grid_sample(feat_per_roi, grid, mode="bilinear", align_corners=True)

        return out


class MultiScaleROIAlign(nn.Module):
    def __init__(
        self,
        output_size: tuple[int, int],
        canonical_level: int = 2,
        canocical_scale: int = 224,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.canonical_level = canonical_level
        self.canonical_scale = canocical_scale

        self.align = ROIAlign(output_size)

    def forward(self, features: list[Tensor], rois: Tensor, roi_idx: Tensor) -> Tensor:
        device = rois.device

        x1, y1, x2, y2 = rois.unbind(axis=-1)
        widths = x2 - x1
        heights = y2 - y1
        scales = lucid.sqrt(widths * heights)

        target_lvls = lucid.log2(scales / self.canonical_scale + 1e-6)
        target_lvls = target_lvls.astype(lucid.Int32) + self.canonical_level
        target_lvls = lucid.clip(target_lvls, 2, 5)

        pooled = []
        for level in range(2, 6):
            mask = (target_lvls == level).nonzero().squeeze(axis=1)
            if mask.size == 0:
                continue

            feat = features[level - 2]
            rois_l = rois[mask]
            idx_l = roi_idx[mask]
            pooled.append(self.align(feat, rois_l, idx_l))

        if pooled:
            return lucid.concatenate(pooled, axis=0)
        else:
            B, C, ph, pw = features[0].shape[0], features[0].shape[1], *self.output_size
            return lucid.empty(B, C, ph, pw, device=device)


class FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_c, out_channels, kernel_size=1) for in_c in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        feats = [lateral(f) for f, lateral in zip(features, self.lateral_convs)]
        for i in reversed(range(len(feats) - 1)):
            up = F.interpolate(feats[i + 1], size=feats[i].shape[2:], mode="nearest")
            feats[i] += up

        return [conv(f) for f, conv in zip(feats, self.output_convs)]
