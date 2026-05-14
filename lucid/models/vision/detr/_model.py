"""DETR — DEtection TRansformer (Carion et al., ECCV 2020).

Paper: "End-to-End Object Detection with Transformers"

Key innovation
--------------
DETR casts object detection as a **direct set prediction** problem:

1. A CNN backbone extracts a spatial feature map.
2. A Transformer encoder attends globally over the (flattened) feature map.
3. A fixed set of N learned *object queries* decode the encoder memory via
   cross-attention in the Transformer decoder.
4. Independent FFN heads map each query to a class distribution and a
   bounding-box tuple (cx, cy, w, h) normalised to [0, 1].
5. At training time, a **Hungarian matching** finds the optimal bijection
   between predictions and ground-truth objects and computes the set loss.
6. At inference, the N predictions are thresholded by class score — no NMS,
   no anchors, no proposal stages.

Architecture
------------
  Image (B, C, H, W)
    ↓  ResNet-50 (conv1 → pool → layer1–layer4) — C5 (2048ch, stride 32)
    ↓  1×1 Conv: 2048 → d_model  (default 256)
    ↓  Flatten: (B, d_model, H', W') → (H'·W', B, d_model)
    ↓  + 2D sinusoidal positional encoding  (B, d_model, H', W') → flat
  Encoder: N_enc × (self-attention + FFN)
  Decoder: N_dec × (self-attention on queries + cross-attention to memory + FFN)
    ↓  (N_queries, B, d_model)
  Class head: Linear → (B, N, num_classes + 1)   softmax at inference
  Box head:   3-layer MLP → (B, N, 4)   sigmoid to enforce [0,1]

Losses (training)
-----------------
  Requires ``targets`` — list of B dicts with:
    ``"boxes"``  : (M_i, 4) xyxy, normalised to [0, 1] by image H/W.
    ``"labels"`` : (M_i,)  integer foreground class ids.

  Hungarian-matched set loss:
    L = λ_cls · L_cls  +  λ_l1 · L_L1  +  λ_giou · L_GIoU
    with λ_cls=1, λ_l1=5, λ_giou=2  (paper §A.4 / Table 10).

  L_cls : cross-entropy on matched pair (background weight = 0.1 for
          unmatched queries).
  L_L1  : ℓ1 between predicted and GT cxcywh boxes (matched pairs only).
  L_GIoU: GIoU loss between decoded and GT boxes (matched pairs only).

Faithfulness notes
------------------
* ResNet-50 C5 feature map (stride 32) with batch-norm.
* d_model=256, n_head=8, 6 enc / 6 dec layers, dim_ffn=2048.
* N=100 object queries.
* Sinusoidal 2-D positional encoding identical to §A.4.
* Hungarian cost matrix: L_cls + 5·L_L1 + 2·L_GIoU (matched queries).
* Background class weight 0.1 for unmatched queries.
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    generalized_box_iou,
)
from lucid.models.vision.detr._config import DETRConfig

# ---------------------------------------------------------------------------
# ResNet-50 backbone (C5 only — same building blocks as Mask R-CNN)
# ---------------------------------------------------------------------------


class _Bottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x
        out: Tensor = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        out = F.relu(cast(Tensor, self.bn2(cast(Tensor, self.conv2(out)))))
        out = cast(Tensor, self.bn3(cast(Tensor, self.conv3(out))))
        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))
        return F.relu(out + identity)


def _make_layer(
    in_ch: int, mid_ch: int, num_blocks: int, stride: int = 1
) -> tuple[nn.Sequential, int]:
    out_ch = mid_ch * 4
    ds: nn.Module | None = None
    if stride != 1 or in_ch != out_ch:
        ds = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    blocks: list[nn.Module] = [_Bottleneck(in_ch, mid_ch, stride=stride, downsample=ds)]
    for _ in range(1, num_blocks):
        blocks.append(_Bottleneck(out_ch, mid_ch))
    return nn.Sequential(*blocks), out_ch


class _ResNet50C5(nn.Module):
    """ResNet-50 backbone, returns C5 feature map (stride 32, 2048ch)."""

    def __init__(self, in_channels: int, layers: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, c2 = _make_layer(64, 64, layers[0], stride=1)
        self.layer2, c3 = _make_layer(c2, 128, layers[1], stride=2)
        self.layer3, c4 = _make_layer(c3, 256, layers[2], stride=2)
        self.layer4, c5 = _make_layer(c4, 512, layers[3], stride=2)
        self.out_channels: int = c5

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        x = cast(Tensor, self.pool(x))
        x = cast(Tensor, self.layer1(x))
        x = cast(Tensor, self.layer2(x))
        x = cast(Tensor, self.layer3(x))
        return cast(Tensor, self.layer4(x))


# ---------------------------------------------------------------------------
# 2-D sinusoidal positional encoding
# ---------------------------------------------------------------------------


def _build_2d_sin_pos_enc(
    h: int,
    w: int,
    d_model: int,
    temperature: float = 10000.0,
    device: str = "cpu",
) -> Tensor:
    """Build 2-D sine / cosine positional encoding of shape (h*w, d_model).

    Each spatial position (row r, col c) gets a d_model-dim embedding:
      dim 0 … d//2-1  : function of column c
      dim d//2 … d-1  : function of row r
    matching the implementation in §A.4 of the DETR paper.
    """
    assert d_model % 2 == 0, "d_model must be even for 2-D positional encoding"
    half = d_model // 2
    quarter = half // 2

    # column encoding (x-axis)
    col_data: list[list[float]] = []
    for c in range(w):
        row: list[float] = []
        for i in range(quarter):
            omega = math.pow(temperature, -2.0 * i / half)
            row.append(math.sin(c * omega))
            row.append(math.cos(c * omega))
        col_data.append(row)
    col_enc = lucid.tensor(col_data, device=device)  # (w, half)

    # row encoding (y-axis)
    row_data: list[list[float]] = []
    for r in range(h):
        row2: list[float] = []
        for i in range(quarter):
            omega = math.pow(temperature, -2.0 * i / half)
            row2.append(math.sin(r * omega))
            row2.append(math.cos(r * omega))
        row_data.append(row2)
    row_enc = lucid.tensor(row_data, device=device)  # (h, half)

    # Tile: (h, w, d_model)
    # For each (r,c): first half = col_enc[c], second half = row_enc[r]
    pos_enc_data: list[list[list[float]]] = []
    for r in range(h):
        row_slice: list[list[float]] = []
        for c in range(w):
            vec: list[float] = []
            for d in range(half):
                vec.append(float(col_enc[c, d].item()))
            for d in range(half):
                vec.append(float(row_enc[r, d].item()))
            row_slice.append(vec)
        pos_enc_data.append(row_slice)

    pos_2d = lucid.tensor(pos_enc_data, device=device)  # (h, w, d_model)
    return pos_2d.reshape(h * w, d_model)  # (h*w, d_model)


# ---------------------------------------------------------------------------
# MLP head (for bounding-box prediction)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Simple N-layer MLP with ReLU activations (last layer has no activation)."""

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.net(x))


# ---------------------------------------------------------------------------
# Hungarian matching (assignment)
# ---------------------------------------------------------------------------


def _hungarian_match(
    pred_logits: Tensor,  # (N, K+1)
    pred_boxes: Tensor,  # (N, 4) cxcywh normalised
    gt_labels: Tensor,  # (M,)
    gt_boxes: Tensor,  # (M, 4) cxcywh normalised
    cost_cls: float = 1.0,
    cost_l1: float = 5.0,
    cost_giou: float = 2.0,
) -> tuple[list[int], list[int]]:
    """Minimum-cost bipartite matching between N queries and M GT objects.

    Uses a simple O(N·M) cost matrix solved by the Hungarian algorithm
    (O(N²M) greedy augmenting-path variant — N is small, typically 100).

    Returns:
        (pred_indices, gt_indices) — matched pairs, in ascending pred order.
    """
    N = int(pred_logits.shape[0])
    M = int(gt_labels.shape[0])
    if M == 0:
        return [], []

    # Classification cost: negative softmax probability for the GT class
    scores = F.softmax(pred_logits, dim=-1)  # (N, K+1)

    # L1 cost: sum of absolute differences in cxcywh space
    # GIoU cost: negative GIoU between decoded boxes
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)  # (M, 4)
    pred_xy = box_cxcywh_to_xyxy(pred_boxes)  # (N, 4)
    giou_mat = generalized_box_iou(pred_xy, gt_xy)  # (N, M)

    # Build cost matrix (N, M) using Python loops over small M
    cost: list[list[float]] = []
    for n in range(N):
        row: list[float] = []
        for m in range(M):
            gt_cls = int(gt_labels[m].item())
            c_cls = -float(scores[n, gt_cls].item())

            # L1
            c_l1 = sum(
                abs(float(pred_boxes[n, d].item()) - float(gt_boxes[m, d].item()))
                for d in range(4)
            )
            # GIoU (negate — lower cost = better)
            c_giou = -float(giou_mat[n, m].item())

            row.append(cost_cls * c_cls + cost_l1 * c_l1 + cost_giou * c_giou)
        cost.append(row)

    # Greedy augmenting-path Hungarian (O(N*M*min(N,M)))
    # Implemented as the classic O(N^2*M) algorithm for correctness
    INF = float("inf")
    # u[n]: potential for query n; v[m]: potential for GT m
    u = [0.0] * (N + 1)
    v = [0.0] * (M + 1)
    p = [0] * (M + 1)  # p[m] = which query is matched to GT m (1-indexed)
    way = [0] * (M + 1)

    for n in range(1, N + 1):
        p[0] = n
        j0 = 0
        minv = [INF] * (M + 1)
        used = [False] * (M + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, M + 1):
                if not used[j]:
                    # cost is 0-indexed: i0-1, j-1
                    c = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if c < minv[j]:
                        minv[j] = c
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if j1 == -1:
                break
            for j in range(M + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    pred_idx: list[int] = []
    gt_idx: list[int] = []
    for m in range(1, M + 1):
        if p[m] != 0:
            pred_idx.append(p[m] - 1)  # 0-indexed
            gt_idx.append(m - 1)

    # Sort by pred_idx ascending
    pairs = sorted(zip(pred_idx, gt_idx), key=lambda t: t[0])
    if pairs:
        pi_sorted, gi_sorted = zip(*pairs)
        return list(pi_sorted), list(gi_sorted)
    return [], []


# ---------------------------------------------------------------------------
# DETR model
# ---------------------------------------------------------------------------


class DETRForObjectDetection(PretrainedModel):
    """DETR end-to-end object detector (Carion et al., ECCV 2020).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch (values normalised to image mean/std).
    ``targets`` : optional training list of B dicts with:
                    ``"boxes"``  — (M_i, 4) xyxy **normalised** to [0,1] by H,W.
                    ``"labels"`` — (M_i,)   integer foreground class ids.

    Output contract
    ---------------
    ``ObjectDetectionOutput``:
      ``logits``    : (B, N, num_classes+1) raw class logits per query.
      ``pred_boxes``: (B, N, 4) cxcywh normalised to [0,1] (sigmoid output).
      ``loss``      : total set-prediction loss when targets provided.
    """

    config_class: ClassVar[type[DETRConfig]] = DETRConfig
    base_model_prefix: ClassVar[str] = "detr"

    def __init__(self, config: DETRConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model

        # Backbone
        self.backbone = _ResNet50C5(config.in_channels, config.backbone_layers)
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d, 1)

        # Object queries
        self.query_embed = nn.Embedding(config.num_queries, d)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d,
            nhead=config.n_head,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Prediction heads
        self.class_embed = nn.Linear(d, config.num_classes + 1)
        self.bbox_embed = _MLP(
            in_dim=d,
            hidden_dim=config.bbox_hidden_dim,
            out_dim=4,
            n_layers=config.num_bbox_layers,
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run DETR.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional training targets (list of dicts with
                     "boxes" (xyxy normalised) and "labels").

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``    : (B, N, K+1) raw class logits.
              ``pred_boxes``: (B, N, 4) cxcywh normalised boxes.
              ``loss``      : set-prediction loss when targets provided.
        """
        B = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # 1. Backbone + projection → (B, d, H', W')
        feat: Tensor = cast(Tensor, self.backbone(x))
        feat = cast(Tensor, self.input_proj(feat))  # (B, d, H', W')

        fH = int(feat.shape[2])
        fW = int(feat.shape[3])
        d = int(feat.shape[1])

        device = feat.device.type

        # 2. 2-D positional encoding → add to feature map
        pos_enc = _build_2d_sin_pos_enc(fH, fW, d, device=device)  # (H'W', d)
        # Expand to (S, B, d) by repeating across batch
        pos_t = pos_enc.unsqueeze(1).expand(-1, B, -1)  # (S, B, d)

        # Flatten feature: (B, d, H'W') → (H'W', B, d) for Transformer
        src = feat.reshape(B, d, fH * fW).permute(2, 0, 1)  # (S, B, d)
        src = src + pos_t

        # 3. Object queries → (N, B, d)
        queries: Tensor = cast(Tensor, self.query_embed.weight)  # (N, d)
        N = int(queries.shape[0])
        tgt = lucid.zeros((N, B, d), device=device)  # (N, B, d) — queries start as zero

        # 4. Transformer (src as memory, tgt as queries)
        # nn.Transformer expects: src=(S, B, d), tgt=(T, B, d)
        pos_queries = queries.unsqueeze(1).expand(-1, B, -1)  # (N, B, d)
        hs: Tensor = cast(
            Tensor,
            self.transformer(
                src=src,
                tgt=tgt + pos_queries,  # add positional query embeddings to tgt
            ),
        )
        # hs: (N, B, d) — decoder output

        # Rearrange to (B, N, d)
        hs_bn = hs.permute(1, 0, 2)  # (B, N, d)

        # 5. Prediction heads
        logits: Tensor = cast(Tensor, self.class_embed(hs_bn))  # (B, N, K+1)
        pred_boxes: Tensor = F.sigmoid(
            cast(Tensor, self.bbox_embed(hs_bn))
        )  # (B, N, 4)

        # 6. Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._set_loss(logits, pred_boxes, targets, (iH, iW))

        return ObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            loss=loss,
        )

    # ------------------------------------------------------------------
    # Set-prediction loss
    # ------------------------------------------------------------------

    def _set_loss(
        self,
        logits: Tensor,
        pred_boxes: Tensor,
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> Tensor:
        """Hungarian-matched set loss across the batch."""
        B = int(logits.shape[0])
        N = int(logits.shape[1])
        K = self._cfg.num_classes
        iH, iW = image_size

        cls_losses: list[Tensor] = []
        l1_losses: list[Tensor] = []
        giou_losses: list[Tensor] = []

        bg_weight = 0.1  # down-weight "no object" in CE loss

        for b in range(B):
            lg_b = logits[b]  # (N, K+1)
            pb_b = pred_boxes[b]  # (N, 4) cxcywh

            gt_boxes_xyxy = targets[b]["boxes"]  # (M, 4) xyxy normalised [0,1]
            gt_labels = targets[b]["labels"]  # (M,)
            M = int(gt_boxes_xyxy.shape[0])

            # Convert GT from xyxy → cxcywh for L1 cost
            gt_boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes_xyxy)  # (M, 4)

            if M == 0:
                # All queries should predict background
                bg_tgt = lucid.zeros((N,))
                cls_losses.append(F.cross_entropy(lg_b, bg_tgt, reduction="mean"))
                continue

            # Hungarian matching
            pred_idx, gt_idx = _hungarian_match(
                lg_b,
                pb_b,
                gt_labels,
                gt_boxes_cxcywh,
                cost_cls=1.0,
                cost_l1=5.0,
                cost_giou=2.0,
            )

            # Build per-query target labels (unmatched → background = 0)
            cls_targets_data: list[int] = [0] * N
            for pi, gi in zip(pred_idx, gt_idx):
                cls_targets_data[pi] = int(gt_labels[gi].item())

            # Weight mask: matched queries get full weight; background = bg_weight
            weight_data: list[float] = [bg_weight] * N
            for pi in pred_idx:
                weight_data[pi] = 1.0

            cls_tgt = lucid.tensor(cls_targets_data)
            weight = lucid.tensor(weight_data)

            # Weighted CE (per-sample weight)
            log_sm = F.log_softmax(lg_b, dim=-1)  # (N, K+1)
            ce_per_n: list[Tensor] = []
            for n in range(N):
                c = cls_targets_data[n]
                ce_per_n.append(-log_sm[n, c] * weight[n])
            cls_losses.append(lucid.cat([l.reshape(1) for l in ce_per_n]).mean())

            if not pred_idx:
                continue

            # L1 loss on matched pairs
            pred_matched_data = [
                [float(pb_b[pi, d].item()) for d in range(4)] for pi in pred_idx
            ]
            gt_matched_data = [
                [float(gt_boxes_cxcywh[gi, d].item()) for d in range(4)]
                for gi in gt_idx
            ]
            pred_matched = lucid.tensor(pred_matched_data)  # (P, 4)
            gt_matched = lucid.tensor(gt_matched_data)  # (P, 4)
            l1_losses.append(lucid.abs(pred_matched - gt_matched).mean())

            # GIoU loss on matched pairs
            pred_xyxy = box_cxcywh_to_xyxy(pred_matched)  # (P, 4)
            gt_xyxy = box_cxcywh_to_xyxy(gt_matched)  # (P, 4)
            giou_mat = generalized_box_iou(pred_xyxy, gt_xyxy)  # (P, P)
            giou_diag_data: list[float] = [
                float(giou_mat[i, i].item()) for i in range(len(pred_idx))
            ]
            giou_diag = lucid.tensor(giou_diag_data)
            giou_losses.append((1.0 - giou_diag).mean())

        cls_l = (
            lucid.cat([l.reshape(1) for l in cls_losses]).mean()
            if cls_losses
            else lucid.zeros((1,))
        )
        l1_l = (
            lucid.cat([l.reshape(1) for l in l1_losses]).mean()
            if l1_losses
            else lucid.zeros((1,))
        )
        giou_l = (
            lucid.cat([l.reshape(1) for l in giou_losses]).mean()
            if giou_losses
            else lucid.zeros((1,))
        )

        return 1.0 * cls_l + 5.0 * l1_l + 2.0 * giou_l

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Filter queries by score threshold and convert to pixel coordinates.

        Args:
            output:      Forward pass output.
            image_sizes: List of (H, W) per image.

        Returns:
            List of per-image result dicts with "boxes" (xyxy pixels),
            "scores", and "labels".
        """
        B = int(output.logits.shape[0])
        N = int(output.logits.shape[1])
        results: list[dict[str, Tensor]] = []

        for b in range(B):
            lg_b = output.logits[b]  # (N, K+1)
            pb_b = output.pred_boxes[b]  # (N, 4) cxcywh [0,1]
            iH, iW = image_sizes[b]

            probs = F.softmax(lg_b, dim=-1)  # (N, K+1)

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for n in range(N):
                # Best non-background class
                best_cls = 0
                best_sc = -1.0
                K_plus_1 = int(probs.shape[1])
                for c in range(1, K_plus_1):
                    sc = float(probs[n, c].item())
                    if sc > best_sc:
                        best_sc = sc
                        best_cls = c

                if best_sc < self._cfg.score_thresh:
                    continue

                # Convert cxcywh [0,1] → xyxy pixels
                cx = float(pb_b[n, 0].item()) * iW
                cy = float(pb_b[n, 1].item()) * iH
                w2 = float(pb_b[n, 2].item()) * iW / 2.0
                h2 = float(pb_b[n, 3].item()) * iH / 2.0
                box_data = [
                    [
                        max(0.0, cx - w2),
                        max(0.0, cy - h2),
                        min(float(iW), cx + w2),
                        min(float(iH), cy + h2),
                    ]
                ]
                keep_boxes.append(lucid.tensor(box_data))
                keep_scores.append(lucid.tensor([[best_sc]]))
                keep_labels.append(lucid.tensor([[float(best_cls)]]))

            if keep_boxes:
                results.append(
                    {
                        "boxes": lucid.cat(keep_boxes, dim=0).squeeze(1),
                        "scores": lucid.cat(keep_scores, dim=0).squeeze(1),
                        "labels": lucid.cat(keep_labels, dim=0).squeeze(1),
                    }
                )
            else:
                results.append(
                    {
                        "boxes": lucid.zeros((0, 4)),
                        "scores": lucid.zeros((0,)),
                        "labels": lucid.zeros((0,)),
                    }
                )
        return results
