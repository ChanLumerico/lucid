from typing import Literal, Self

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class _SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = ...  # TODO: implement `nn.AdaptiveMaxPoolNd`
        self.output_size = output_size

    def forward(self, images, rois, roi_idx):
        N = rois.shape[0]
        H, W = images.shape[2:]

        x1, x2 = rois[:, 0], rois[:, 2]
        y1, y2 = rois[:, 1], rois[:, 3]

        x1 = lucid.floor(x1 * W).astype(lucid.Int)
        x2 = lucid.ceil(x2 * W).astype(lucid.Int)
        y1 = lucid.floor(y1 * H).astype(lucid.Int)
        y2 = lucid.ceil(y2 * H).astype(lucid.Int)

        res = []
        for i in range(N):
            img = images[roi_idx[i]].unsqueeze(axis=0)
            img = img[:, :, y1[i] : y2[i], x1[i] : x2[i]]
            ...
