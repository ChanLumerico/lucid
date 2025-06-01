from typing import Literal, Tuple

import lucid
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


@lucid.no_grad()
def _compute_edges(
    image: Tensor, connectivity: Literal[4, 8] = 8
) -> Tuple[Tensor, Tensor, Tensor]:
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
        lucid.concatenate(edges_p),
        lucid.concatenate(edges_q),
        lucid.concatenate(edges_w),
    )


@lucid.no_grad()
def felzenszwalb_segmentation(
    image: Tensor, k: float = 500.0, min_size: int = 20, connectivity: Literal[4, 8] = 8
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("Image must have shape (C, H, W)")

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
