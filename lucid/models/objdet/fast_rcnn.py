from typing import Literal, Self

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class _SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
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
            img = self.maxpool(img)
            res.append(img)

        res = lucid.concatenate(res, axis=0)
        return res


class FastRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_channels: int,
        num_classes: int,
        pool_size: tuple[int, int] = (7, 7),
        hidden_dim: int = 4096,
        bbox_reg_means: tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        bbox_reg_stds: tuple[int, ...] = (0.1, 0.1, 0.2, 0.2),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.roipool = _SlowROIPool(output_size=pool_size)

        self.fc1 = nn.Linear(feat_channels * pool_size[0] * pool_size[1], hidden_dim)
        self.fc2 - nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

        self.bbox_reg_means = Tensor(bbox_reg_means, dtype=lucid.Float16)
        self.bbox_reg_stds = Tensor(bbox_reg_stds, dtype=lucid.Float16)

    def forward(
        self,
        images: Tensor,
        rois: Tensor,
        roi_idx: Tensor,
        *,
        return_feats: bool = False
    ) -> tuple[Tensor, ...]:
        features = self.backbone(images)
        pooled = self.roipool(features, rois, roi_idx)

        N = pooled.shape[0]
        x = pooled.reshape(N, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        if return_feats:
            return cls_logits, bbox_deltas, features
        return cls_logits, bbox_deltas

    ...
