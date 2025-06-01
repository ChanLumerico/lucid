from typing import Literal

import lucid
from lucid._tensor import Tensor


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = lucid.arange(n, dtype=lucid.Int32)
        self.size = lucid.ones(n, dtype=lucid.Int32)
        self.int_diff = lucid.zeros(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[x] != x:
            next = self.parent[x]
            self.parent[x] = root
            x = next

        return root

    def union(self, x: int, y: int, weight: float) -> int:
        x_root, y_root = self.find(x), self.find(y)
        if x_root == y_root:
            return x_root

        if self.size[x_root] < self.size[y_root]:
            x_root, y_root = y_root, x_root

        self.parent[y_root] = x_root
        self.size[x_root] += self.size[y_root]
        self.int_diff[x_root] = max(
            self.int_diff[x_root], self.int_diff[y_root], weight
        )
        return x_root

    def component_size(self, x: int) -> int:
        return self.size[self.find(x)]


@lucid.no_grad()
def _compute_edges(
    image: Tensor, connectivity: Literal[4, 8] = 8
) -> tuple[Tensor, Tensor, Tensor]:
    H, W = image.shape[:2]
    idx = lucid.arange(H * W, dtype=lucid.Int32).reshape(H, W)

    def _color_dist(a: Tensor, b: Tensor) -> Tensor:
        diff = a.astype(lucid.Float) - b.astype(lucid.Float)
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
            a=image[max(0, dy) : H - max(0, -dy), max(0, dx) : W - max(0, -dx)],
            b=image[max(0, -dy) : H - max(0, dy), max(0, -dx) : W - max(0, dx)],
        ).ravel()

        edges_p.append(p)
        edges_q.append(q)
        edges_w.append(w)

    return (
        lucid.concatenate(edges_p),
        lucid.concatenate(edges_q),
        lucid.concatenate(edges_w),
    )


def felzenszwalb_segmentation(
    image: Tensor, k: float = 500.0, min_size: int = 20, connectivity: Literal[4, 8] = 8
) -> Tensor:
    if image.ndim != 3:
        raise ValueError("Image must have shape (C, H, W)")

    C, H, W = image.shape
    img_f32 = image.astype(lucid.Float32)

    if C == 1:
        img_cl = img_f32[0]
    else:
        img_cl = img_f32.transpose((1, 2, 0))

    n_px = H * W
    p, q, w = _compute_edges(img_cl, connectivity)

    order = ...  # TODO: implement `argsort`
