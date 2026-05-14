"""MaskFormer (Cheng et al., NeurIPS 2021).

Paper: "Per-Pixel Classification is Not All You Need for Semantic Segmentation"

Key innovation
--------------
MaskFormer reformulates semantic segmentation as **mask classification**:

1. A CNN backbone + FPN pixel decoder extracts per-pixel embeddings.
2. N learnable queries attend to pixel embeddings via a Transformer decoder.
3. Two prediction heads per query:
   - Class head: Linear(d_model, K+1) — class distribution (K+1 for no-object).
   - Mask head: dot product of query embedding with per-pixel embeddings →
     binary mask logits (B, N, H/4, W/4).
4. Training: Hungarian matching (mask IoU + cross-entropy) + BCE mask loss +
   CE class loss on matched pairs.
5. Inference: weighted sum of class probs × mask probs, upsampled to (H, W).

Architecture
------------
  Image (B, C, H, W)
    ↓  ResNet backbone → [C2(256), C3(512), C4(1024), C5(2048)]
    ↓  FPN lateral convs → [P2, P3, P4, P5] all at fpn_out_channels
    ↓  Pixel Decoder: sum/upsample to P2 resolution (H/4, W/4), project to d_model
  Memory: (B, d_model, H/4, W/4) → flatten → (H/4·W/4, B, d_model)
  Queries: N × d_model — fed as tgt to TransformerDecoder
  Class head: (B, N, K+1) softmax at inference
  Mask head: dot(query, pixel_embed) → (B, N, H/4, W/4) sigmoid at inference
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models._registry import register_model
from lucid.models.vision.maskformer._config import MaskFormerConfig

# ---------------------------------------------------------------------------
# ResNet backbone (C2–C5 feature maps)
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    """ResNet BasicBlock (used by ResNet-18 / -34): 3×3 → 3×3, expansion 1."""

    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(
            in_ch,
            mid_ch,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            out_ch,
            3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x
        out: Tensor = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        out = cast(Tensor, self.bn2(cast(Tensor, self.conv2(out))))
        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))
        return F.relu(out + identity)


class _Bottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            mid_ch,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
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
    in_ch: int,
    mid_ch: int,
    num_blocks: int,
    stride: int = 1,
    dilation: int = 1,
    block: type[_BasicBlock] | type[_Bottleneck] = _Bottleneck,
) -> tuple[nn.Sequential, int]:
    out_ch = mid_ch * block.expansion
    ds: nn.Module | None = None
    if stride != 1 or in_ch != out_ch:
        ds = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    blocks: list[nn.Module] = [
        block(in_ch, mid_ch, stride=stride, downsample=ds, dilation=dilation)
    ]
    for _ in range(1, num_blocks):
        blocks.append(block(out_ch, mid_ch, dilation=dilation))
    return nn.Sequential(*blocks), out_ch


class _ResNetBackbone(nn.Module):
    """ResNet backbone returning [C2, C3, C4, C5] feature maps."""

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
        block_type: str = "bottleneck",
    ) -> None:
        super().__init__()
        block: type[_BasicBlock] | type[_Bottleneck] = (
            _BasicBlock if block_type == "basic" else _Bottleneck
        )
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1, c2 = _make_layer(64, 64, layers[0], stride=1, block=block)
        self.layer2, c3 = _make_layer(c2, 128, layers[1], stride=2, block=block)
        self.layer3, c4 = _make_layer(c3, 256, layers[2], stride=2, block=block)
        self.layer4, c5 = _make_layer(c4, 512, layers[3], stride=2, block=block)
        self.out_channels: list[int] = [c2, c3, c4, c5]

    def forward(self, x: Tensor) -> list[Tensor]:  # type: ignore[override]
        x = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        x = cast(Tensor, self.pool(x))
        c2: Tensor = cast(Tensor, self.layer1(x))
        c3: Tensor = cast(Tensor, self.layer2(c2))
        c4: Tensor = cast(Tensor, self.layer3(c3))
        c5: Tensor = cast(Tensor, self.layer4(c4))
        return [c2, c3, c4, c5]


# ---------------------------------------------------------------------------
# FPN Pixel Decoder
# ---------------------------------------------------------------------------


class _FPNPixelDecoder(nn.Module):
    """Feature Pyramid Network pixel decoder.

    Takes [C2, C3, C4, C5] from the backbone and produces a single
    per-pixel embedding map at 1/4 resolution projected to d_model channels.
    """

    def __init__(
        self,
        in_channels: list[int],  # [c2, c3, c4, c5]
        fpn_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()
        # Lateral 1x1 convs for each scale
        self.lat5 = nn.Conv2d(in_channels[3], fpn_ch, 1)
        self.lat4 = nn.Conv2d(in_channels[2], fpn_ch, 1)
        self.lat3 = nn.Conv2d(in_channels[1], fpn_ch, 1)
        self.lat2 = nn.Conv2d(in_channels[0], fpn_ch, 1)
        # 3x3 output convs
        self.out5 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.out4 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.out3 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.out2 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        # Project to d_model for transformer
        self.proj = nn.Conv2d(fpn_ch, out_ch, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, features: list[Tensor]) -> Tensor:  # type: ignore[override]
        """
        Args:
            features: [C2, C3, C4, C5] from backbone.

        Returns:
            (B, out_ch, H/4, W/4) per-pixel embeddings.
        """
        c2, c3, c4, c5 = features

        # Top-down FPN with per-level 3×3 smoothing (paper §3.2):
        # each lateral is smoothed via its own out_* conv before merging.
        p5: Tensor = cast(Tensor, self.out5(F.relu(cast(Tensor, self.lat5(c5)))))
        p4: Tensor = cast(Tensor, self.out4(
            F.relu(cast(Tensor, self.lat4(c4))) + cast(Tensor, self.up(p5))
        ))
        p3: Tensor = cast(Tensor, self.out3(
            F.relu(cast(Tensor, self.lat3(c3))) + cast(Tensor, self.up(p4))
        ))
        p2: Tensor = cast(Tensor, self.out2(
            F.relu(cast(Tensor, self.lat2(c2))) + cast(Tensor, self.up(p3))
        ))

        # Project to d_model
        return cast(Tensor, self.proj(p2))  # (B, out_ch, H/4, W/4)


# ---------------------------------------------------------------------------
# Hungarian matching for semantic segmentation masks
# ---------------------------------------------------------------------------


def _binary_mask_iou(pred_mask: Tensor, gt_mask: Tensor) -> float:
    """Compute IoU between two binary masks (after sigmoid/threshold).

    Vectorised: single ``.sum().item()`` per call (no per-pixel sync).

    Args:
        pred_mask: (H, W) float predictions in [0, 1].
        gt_mask:   (H, W) binary ground truth.

    Returns:
        IoU as a float.
    """
    p_bin = (pred_mask > 0.5).float()
    g_bin = (gt_mask > 0.5).float()
    inter = float((p_bin * g_bin).sum().item())
    union = float((p_bin + g_bin - p_bin * g_bin).sum().item())
    if union < 1e-6:
        return 1.0 if inter < 1e-6 else 0.0
    return inter / union


def _hungarian_match_masks(
    class_logits: Tensor,  # (N, K+1)
    mask_logits: Tensor,  # (N, H, W)
    gt_labels: Tensor,  # (M,) integer class ids
    gt_masks: Tensor,  # (M, H, W) binary
) -> tuple[list[int], list[int]]:
    """Hungarian matching between N queries and M GT segments.

    Cost: -log_prob(gt_class) - mask_iou(sigmoid(pred), gt).

    Returns:
        (pred_indices, gt_indices) matched pairs.
    """
    N = int(class_logits.shape[0])
    M = int(gt_labels.shape[0])
    if M == 0 or N == 0:
        return [], []

    probs = F.softmax(class_logits, dim=-1)  # (N, K+1)
    pred_probs = F.sigmoid(mask_logits)  # (N, H, W)

    # Build M × N cost matrix (rows = GTs, cols = queries — M ≤ N)
    cost: list[list[float]] = []
    for m in range(M):
        gt_cls = int(gt_labels[m].item())
        row: list[float] = []
        for n in range(N):
            c_cls = -float(probs[n, gt_cls].item())
            iou = _binary_mask_iou(pred_probs[n], gt_masks[m])
            row.append(c_cls - iou)
        cost.append(row)

    from lucid.models.vision.detr._model import _kuhn_munkres_rectangular
    gt_idx, pred_idx = _kuhn_munkres_rectangular(cost)
    return pred_idx, gt_idx


# ---------------------------------------------------------------------------
# MaskFormer model
# ---------------------------------------------------------------------------


class MaskFormerForSemanticSegmentation(PretrainedModel):
    """MaskFormer semantic segmentation model (Cheng et al., NeurIPS 2021).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch.
    ``targets`` : optional dict with ``"masks"`` key — (B, H, W) integer
                  segmentation labels — for computing training loss.

    Output contract
    ---------------
    ``SemanticSegmentationOutput``:
      ``logits`` : (B, num_classes+1, H, W) — per-pixel class logits
                   (weighted combination of query predictions).
      ``loss``   : Hungarian-matched BCE mask + CE class loss when targets
                   provided.
    """

    config_class: ClassVar[type[MaskFormerConfig]] = MaskFormerConfig
    base_model_prefix: ClassVar[str] = "maskformer"

    def __init__(self, config: MaskFormerConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model
        K = config.num_classes

        # 1. Backbone
        self.backbone = _ResNetBackbone(
            config.in_channels,
            config.backbone_layers,
            config.backbone_block,
        )

        # 2. Pixel decoder
        self.pixel_decoder = _FPNPixelDecoder(
            self.backbone.out_channels,
            fpn_ch=config.fpn_out_channels,
            out_ch=d,
        )

        # 3. Query embeddings
        self.query_embed = nn.Embedding(config.num_queries, d)

        # 4. Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=config.n_head,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=config.num_decoder_layers,
        )

        # 5. Prediction heads
        self.class_head = nn.Linear(d, K + 1)
        self.mask_embed = nn.Linear(d, d)

        # 6. Per-pixel embedding projection (used in mask head dot product)
        self.pixel_proj = nn.Conv2d(d, d, 1)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: dict[str, Tensor] | None = None,
    ) -> SemanticSegmentationOutput:
        """Run MaskFormer.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional dict with ``"masks"`` (B, H, W) integer masks.

        Returns:
            ``SemanticSegmentationOutput``:
              ``logits``: (B, K+1, H, W) per-pixel class logits.
              ``loss``:   training loss when targets provided.
        """
        B = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # 1. Backbone + pixel decoder → (B, d, H/4, W/4)
        features: list[Tensor] = self.backbone.forward(x)
        pixel_emb: Tensor = self.pixel_decoder.forward(features)
        pixel_emb = cast(Tensor, self.pixel_proj(pixel_emb))

        fH = int(pixel_emb.shape[2])
        fW = int(pixel_emb.shape[3])
        d = int(pixel_emb.shape[1])

        # 2. Flatten pixel embeddings → (S, B, d) for transformer
        # pixel_emb: (B, d, H, W) → (B, d, H*W) → (H*W, B, d)
        mem: Tensor = pixel_emb.reshape(B, d, fH * fW).permute(2, 0, 1)

        # 3. Object queries → (N, B, d)
        N = self._cfg.num_queries
        q_embed: Tensor = cast(Tensor, self.query_embed.weight)  # (N, d)
        tgt: Tensor = q_embed.unsqueeze(1).expand(-1, B, -1)  # (N, B, d)

        # 4. Transformer decoder
        hs: Tensor = self.transformer_decoder.forward(tgt, mem)  # (N, B, d)

        # 5. Rearrange → (B, N, d)
        hs_bn: Tensor = hs.permute(1, 0, 2)

        # 6. Class predictions: (B, N, K+1)
        class_logits: Tensor = cast(Tensor, self.class_head(hs_bn))

        # 7. Mask predictions: (B, N, H/4, W/4)
        mask_embed: Tensor = cast(Tensor, self.mask_embed(hs_bn))  # (B, N, d)
        # pixel_emb: (B, d, H/4, W/4) → (B, d, H*W) → dot with (B, N, d)
        pixel_flat: Tensor = pixel_emb.reshape(B, d, fH * fW)  # (B, d, S)
        # (B, N, d) × (B, d, S) → (B, N, S)
        mask_logits_flat: Tensor = lucid.matmul(mask_embed, pixel_flat)
        mask_logits: Tensor = mask_logits_flat.reshape(B, N, fH, fW)

        # 8. Compute output seg logits (inference)
        # class_probs: (B, N, K+1), mask_probs: (B, N, H/4, W/4)
        class_probs: Tensor = F.softmax(class_logits, dim=-1)  # (B, N, K+1)
        mask_probs: Tensor = F.sigmoid(mask_logits)  # (B, N, H/4, W/4)

        K_plus_1 = self._cfg.num_classes + 1
        # seg_logits[b,k,h,w] = sum_n class_probs[b,n,k] * mask_probs[b,n,h,w]
        # Reshape for bmm: class_probs (B, K+1, N) × mask_probs (B, N, H*W)
        class_probs_t: Tensor = class_probs.permute(0, 2, 1)  # (B, K+1, N)
        mask_flat: Tensor = mask_probs.reshape(B, N, fH * fW)  # (B, N, S)
        seg_flat: Tensor = lucid.matmul(class_probs_t, mask_flat)  # (B, K+1, S)
        seg_logits_small: Tensor = seg_flat.reshape(B, K_plus_1, fH, fW)

        # Upsample to input resolution
        seg_logits: Tensor = F.interpolate(
            seg_logits_small, size=(iH, iW), mode="bilinear", align_corners=False
        )

        # 9. Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(class_logits, mask_logits, targets, (fH, fW))

        return SemanticSegmentationOutput(logits=seg_logits, loss=loss)

    def _compute_loss(
        self,
        class_logits: Tensor,  # (B, N, K+1)
        mask_logits: Tensor,  # (B, N, H/4, W/4)
        targets: dict[str, Tensor],
        feat_size: tuple[int, int],
    ) -> Tensor:
        """Compute Hungarian-matched mask + class loss across batch."""
        B = int(class_logits.shape[0])
        N = int(class_logits.shape[1])
        K = self._cfg.num_classes
        fH, fW = feat_size

        gt_masks_full: Tensor = targets["masks"]  # (B, H, W) integer

        cls_losses: list[Tensor] = []
        mask_losses: list[Tensor] = []

        for b in range(B):
            cl_b: Tensor = class_logits[b]  # (N, K+1)
            ml_b: Tensor = mask_logits[b]  # (N, H/4, W/4)
            seg_b: Tensor = gt_masks_full[b]  # (H, W) integer

            # Discover unique classes present (exclude 0 if it's background)
            unique_classes: list[int] = []
            seg_H = int(seg_b.shape[0])
            seg_W = int(seg_b.shape[1])
            seen: set[int] = set()
            for h in range(seg_H):
                for w in range(seg_W):
                    c = int(seg_b[h, w].item())
                    if c not in seen:
                        seen.add(c)
                        unique_classes.append(c)

            M = len(unique_classes)

            if M == 0:
                # All background
                bg_tgt_data: list[int] = [K] * N  # "no-object" class index = K
                bg_tgt: Tensor = lucid.tensor(bg_tgt_data)
                cls_losses.append(F.cross_entropy(cl_b, bg_tgt, reduction="mean"))
                continue

            # Build GT binary masks for each class at feature resolution
            gt_label_data: list[int] = unique_classes
            gt_masks_list: list[list[list[float]]] = []
            for cls in unique_classes:
                row_data: list[list[float]] = []
                for h in range(seg_H):
                    r: list[float] = []
                    for w in range(seg_W):
                        r.append(1.0 if int(seg_b[h, w].item()) == cls else 0.0)
                    row_data.append(r)
                gt_masks_list.append(row_data)

            gt_labels_t: Tensor = lucid.tensor(gt_label_data)  # (M,)
            gt_masks_t: Tensor = lucid.tensor(gt_masks_list)  # (M, H, W)

            # Downsample GT masks to feature resolution for matching
            gt_masks_small: Tensor = F.interpolate(
                gt_masks_t.reshape(M, 1, seg_H, seg_W),
                size=(fH, fW),
                mode="nearest",
            ).reshape(M, fH, fW)

            # Hungarian match
            pred_idx, gt_idx = _hungarian_match_masks(
                cl_b,
                ml_b.detach() if hasattr(ml_b, "detach") else ml_b,
                gt_labels_t,
                gt_masks_small,
            )

            # Class loss: matched → gt class, unmatched → K (no-object)
            cls_tgt_data: list[int] = [K] * N
            for pi, gi in zip(pred_idx, gt_idx):
                cls_tgt_data[pi] = int(gt_labels_t[gi].item())
            cls_tgt: Tensor = lucid.tensor(cls_tgt_data)
            cls_losses.append(F.cross_entropy(cl_b, cls_tgt, reduction="mean"))

            # Mask loss: BCE on matched pairs
            if pred_idx:
                for pi, gi in zip(pred_idx, gt_idx):
                    pred_m: Tensor = ml_b[pi]  # (H/4, W/4)
                    gt_m: Tensor = gt_masks_small[gi]  # (H/4, W/4)
                    mask_losses.append(
                        F.binary_cross_entropy_with_logits(
                            pred_m.reshape(-1),
                            gt_m.reshape(-1),
                        )
                    )

        cls_loss: Tensor = (
            lucid.cat([l.reshape(1) for l in cls_losses]).mean()
            if cls_losses
            else lucid.zeros((1,))
        )
        mask_loss: Tensor = (
            lucid.cat([l.reshape(1) for l in mask_losses]).mean()
            if mask_losses
            else lucid.zeros((1,))
        )

        return cls_loss + mask_loss
