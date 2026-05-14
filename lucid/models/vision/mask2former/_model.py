"""Mask2Former (Cheng et al., CVPR 2022).

Paper: "Masked-attention Mask Transformer for Universal Image Segmentation"

Key improvements over MaskFormer
---------------------------------
1. **Masked cross-attention (MCA)**: each transformer decoder layer limits
   query-to-memory attention to the predicted mask region from the prior
   layer.  This focuses each query on its candidate region and prevents
   distraction from background pixels.

2. **Multi-scale features**: the pixel decoder produces FPN feature levels
   {P3, P4, P5}.  Consecutive decoder layers attend to different FPN levels
   (cycling: level_idx = layer_idx % num_feature_levels).

3. **Improved pixel decoder**: multi-output FPN with per-level 3×3 output
   convs; P2 (finest) projected to d_model for the mask head dot product.

Architecture
------------
  Image (B, C, H, W)
    ↓  ResNet backbone → [C2, C3, C4, C5]
    ↓  Multi-scale FPN → [P3, P4, P5] (B, fpn_ch, H/8, W/8–W/32)
                        + P2 → pixel_emb (B, d_model, H/4, W/4)
  Decoder layers (num_decoder_layers):
    - Receive tgt queries: (N, B, d)
    - MCA: limit cross-attn to foreground mask from prev layer
    - Cross-attend to FPN level l = layer_idx % num_feature_levels
    - Self-attend among queries
    - FFN
  Class head: (B, N, K+1)
  Mask head: dot(query, pixel_emb) → (B, N, H/4, W/4)
  Seg output: weighted combination, upsample to (H, W)

Losses
------
  Same Hungarian matching + BCE mask + CE class as MaskFormer.
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
from lucid.models.vision.mask2former._config import Mask2FormerConfig

# ---------------------------------------------------------------------------
# ResNet backbone (self-contained — no cross-family import)
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


class _SwinBackbone(nn.Module):
    """Swin Transformer backbone returning [C2, C3, C4, C5] in (B, C, H, W) format.

    Strides: patch_size=4 → C2 at stride 4, then C3/C4/C5 at strides 8/16/32.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 96,
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        from lucid.models.vision.swin._model import _PatchEmbed, _SwinStage

        super().__init__()
        self.patch_embed = _PatchEmbed(in_channels, patch_size=4, embed_dim=embed_dim)
        stages: list[nn.Module] = []
        dim = embed_dim
        out_channels: list[int] = []
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            downsample = i < len(depths) - 1
            stages.append(
                _SwinStage(
                    dim=dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                    downsample=downsample,
                )
            )
            out_channels.append(dim)
            if downsample:
                dim *= 2
        self.stages = nn.ModuleList(stages)
        self.out_channels: list[int] = out_channels

    def forward(self, x: Tensor) -> list[Tensor]:  # type: ignore[override]
        # PatchEmbed → (B, H/4, W/4, C)
        h: Tensor = cast(Tensor, self.patch_embed(x))
        feats: list[Tensor] = []
        from lucid.models.vision.swin._model import _SwinStage
        for stage_mod in self.stages:
            # _SwinStage.forward applies blocks then (optionally) downsample.
            # We want the feature BEFORE downsample as the FPN tap for level i.
            stage = cast(_SwinStage, stage_mod)
            pre: Tensor = h
            blocks_mod = stage.blocks
            for blk in blocks_mod:
                pre = cast(Tensor, blk(pre))
            feats.append(pre.permute(0, 3, 1, 2))  # (B, C, H, W)
            ds = stage.downsample
            h = cast(Tensor, ds(pre)) if ds is not None else pre
        return feats


# ---------------------------------------------------------------------------
# Multi-scale FPN Pixel Decoder
# ---------------------------------------------------------------------------


class _MultiScaleFPNDecoder(nn.Module):
    """FPN pixel decoder producing multi-scale outputs + per-pixel embedding.

    Outputs:
        fpn_levels: list of [P3, P4, P5] tensors at fpn_ch channels.
        pixel_emb: P2-resolution feature projected to out_ch (d_model).
    """

    def __init__(
        self,
        in_channels: list[int],  # [c2, c3, c4, c5]
        fpn_ch: int,
        out_ch: int,
    ) -> None:
        super().__init__()
        self.lat5 = nn.Conv2d(in_channels[3], fpn_ch, 1)
        self.lat4 = nn.Conv2d(in_channels[2], fpn_ch, 1)
        self.lat3 = nn.Conv2d(in_channels[1], fpn_ch, 1)
        self.lat2 = nn.Conv2d(in_channels[0], fpn_ch, 1)
        self.out5 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.out4 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.out3 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.out2 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        # Project finest scale to d_model for mask dot product
        self.proj = nn.Conv2d(fpn_ch, out_ch, 1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(  # type: ignore[override]
        self, features: list[Tensor]
    ) -> tuple[list[Tensor], Tensor]:
        """
        Returns:
            fpn_levels: [P5, P4, P3] (coarsest to finest at fpn_ch).
            pixel_emb:  (B, out_ch, H/4, W/4) per-pixel embeddings.
        """
        c2, c3, c4, c5 = features

        p5: Tensor = F.relu(cast(Tensor, self.lat5(c5)))
        p4: Tensor = F.relu(cast(Tensor, self.lat4(c4))) + cast(Tensor, self.up(p5))
        p3: Tensor = F.relu(cast(Tensor, self.lat3(c3))) + cast(Tensor, self.up(p4))
        p2: Tensor = F.relu(cast(Tensor, self.lat2(c2))) + cast(Tensor, self.up(p3))

        p5_out: Tensor = F.relu(cast(Tensor, self.out5(p5)))
        p4_out: Tensor = F.relu(cast(Tensor, self.out4(p4)))
        p3_out: Tensor = F.relu(cast(Tensor, self.out3(p3)))
        p2_out: Tensor = F.relu(cast(Tensor, self.out2(p2)))

        pixel_emb: Tensor = cast(Tensor, self.proj(p2_out))
        # FPN levels for multi-scale cross-attention: [P3, P4, P5] (finest→coarsest)
        fpn_levels: list[Tensor] = [p3_out, p4_out, p5_out]
        return fpn_levels, pixel_emb


# ---------------------------------------------------------------------------
# Masked-attention decoder layer
# ---------------------------------------------------------------------------


class _MaskedAttentionDecoderLayer(nn.Module):
    """Single Mask2Former decoder layer with masked cross-attention.

    Query cross-attention is restricted to predicted foreground mask pixels.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        fpn_ch: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        # Masked cross-attention: query attends to FPN-level memory
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, kdim=fpn_ch, vdim=fpn_ch
        )
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        # Project FPN feature level to d_model for cross-attn key/value
        self.kv_proj = nn.Linear(fpn_ch, d_model)

    def forward(  # type: ignore[override]
        self,
        tgt: Tensor,  # (N, B, d_model)
        memory: Tensor,  # (S, B, fpn_ch) — FPN level flattened
        attn_mask: Tensor | None = None,  # (N, S) or (B*n_head, N, S) for masking
    ) -> Tensor:
        """One masked-attention decoder layer.

        Args:
            tgt:       (N, B, d) query tensor.
            memory:    (S, B, fpn_ch) current FPN level feature.
            attn_mask: optional (N, S) binary mask — True blocks attention.

        Returns:
            (N, B, d) updated query tensor.
        """
        # Project memory key/value
        mem_d: Tensor = cast(Tensor, self.kv_proj(memory))  # (S, B, d)

        # 1. Masked cross-attention
        tgt2, _ = self.cross_attn(
            tgt,
            mem_d,
            mem_d,
            attn_mask=attn_mask,
            need_weights=False,
        )
        tgt = cast(Tensor, self.norm1(tgt + cast(Tensor, self.dropout1(tgt2))))

        # 2. Self-attention
        tgt2_sa, _ = self.self_attn(tgt, tgt, tgt, need_weights=False)
        tgt = cast(Tensor, self.norm2(tgt + cast(Tensor, self.dropout2(tgt2_sa))))

        # 3. FFN
        ff: Tensor = cast(
            Tensor,
            self.linear2(
                cast(Tensor, self.dropout3(F.relu(cast(Tensor, self.linear1(tgt)))))
            ),
        )
        tgt = cast(Tensor, self.norm3(tgt + cast(Tensor, self.dropout4(ff))))
        return tgt


# ---------------------------------------------------------------------------
# Hungarian matching (same as MaskFormer — self-contained copy)
# ---------------------------------------------------------------------------


def _binary_mask_iou(pred_mask: Tensor, gt_mask: Tensor) -> float:
    """Vectorised binary mask IoU (no per-pixel ``.item()`` sync)."""
    p_bin = (pred_mask > 0.5).float()
    g_bin = (gt_mask > 0.5).float()
    inter = float((p_bin * g_bin).sum().item())
    union = float((p_bin + g_bin - p_bin * g_bin).sum().item())
    if union < 1e-6:
        return 1.0 if inter < 1e-6 else 0.0
    return inter / union


def _hungarian_match_masks(
    class_logits: Tensor,
    mask_logits: Tensor,
    gt_labels: Tensor,
    gt_masks: Tensor,
) -> tuple[list[int], list[int]]:
    N = int(class_logits.shape[0])
    M = int(gt_labels.shape[0])
    if M == 0 or N == 0:
        return [], []

    probs = F.softmax(class_logits, dim=-1)
    pred_probs = F.sigmoid(mask_logits)

    # M × N cost matrix (rows = GTs, cols = queries)
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
# Mask2Former model
# ---------------------------------------------------------------------------


class Mask2FormerForSemanticSegmentation(PretrainedModel):
    """Mask2Former semantic segmentation model (Cheng et al., CVPR 2022).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch.
    ``targets`` : optional dict with ``"masks"`` (B, H, W) integer masks.

    Output contract
    ---------------
    ``SemanticSegmentationOutput``:
      ``logits`` : (B, num_classes+1, H, W) — per-pixel class logits.
      ``loss``   : Hungarian-matched BCE mask + CE class loss when targets
                   provided.
    """

    config_class: ClassVar[type[Mask2FormerConfig]] = Mask2FormerConfig
    base_model_prefix: ClassVar[str] = "mask2former"

    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model
        fpn_ch = config.fpn_out_channels
        K = config.num_classes

        # 1. Backbone (ResNet or Swin)
        self.backbone: _ResNetBackbone | _SwinBackbone
        if config.backbone_type == "swin":
            sd = config.swin_depths
            sh = config.swin_num_heads
            assert len(sd) == 4 and len(sh) == 4, "Swin needs 4-stage depths/num_heads"
            self.backbone = _SwinBackbone(
                in_channels=config.in_channels,
                embed_dim=config.swin_embed_dim,
                depths=(sd[0], sd[1], sd[2], sd[3]),
                num_heads=(sh[0], sh[1], sh[2], sh[3]),
                window_size=config.swin_window_size,
            )
        else:
            self.backbone = _ResNetBackbone(
                config.in_channels,
                config.backbone_layers,
                config.backbone_block,
            )

        # 2. Multi-scale pixel decoder
        self.pixel_decoder = _MultiScaleFPNDecoder(
            self.backbone.out_channels,
            fpn_ch=fpn_ch,
            out_ch=d,
        )

        # 3. Query embeddings
        self.query_embed = nn.Embedding(config.num_queries, d)

        # 4. Masked-attention decoder layers
        self.decoder_layers: list[_MaskedAttentionDecoderLayer] = []
        for i in range(config.num_decoder_layers):
            layer = _MaskedAttentionDecoderLayer(
                d_model=d,
                n_head=config.n_head,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                fpn_ch=fpn_ch,
            )
            self.add_module(f"dec_layer_{i}", layer)
            self.decoder_layers.append(layer)

        # 5. Prediction heads
        self.class_head = nn.Linear(d, K + 1)
        self.mask_embed = nn.Linear(d, d)
        self.pixel_proj = nn.Conv2d(d, d, 1)

        # 6. Auxiliary mask head for computing attention masks between layers
        self.aux_mask_norm = nn.LayerNorm(d)

    def _compute_mask_attn(
        self,
        queries: Tensor,  # (N, B, d)
        pixel_emb: Tensor,  # (B, d, H/4, W/4)
        mem_H: int,
        mem_W: int,
    ) -> Tensor:
        """Compute binary attention mask for masked cross-attention.

        Args:
            queries:   (N, B, d) current query embeddings.
            pixel_emb: (B, d, H/4, W/4) pixel embedding (at P2 resolution).
            mem_H, mem_W: spatial dims of current FPN level memory.

        Returns:
            (N, B * n_head, 1, S_fpn) attention bias — large negative where
            masked out, 0 elsewhere.  Actually returns (N*B, 1, S) for bmm.
            We return (N, S_fpn) boolean mask for passing as attn_mask.
            Returned shape: (N, mem_H*mem_W) — True = block.
        """
        N = int(queries.shape[0])
        B = int(queries.shape[1])
        d = int(queries.shape[2])

        # Compute per-query mask at pixel resolution
        q_t: Tensor = queries.permute(1, 0, 2)  # (B, N, d)
        mask_emb: Tensor = cast(Tensor, self.mask_embed(q_t))  # (B, N, d)
        p_flat: Tensor = pixel_emb.reshape(
            B, d, int(pixel_emb.shape[2]) * int(pixel_emb.shape[3])
        )  # (B, d, S_p2)
        mask_logits: Tensor = lucid.matmul(mask_emb, p_flat)  # (B, N, S_p2)

        # Downsample to current FPN level resolution
        S_fpn = mem_H * mem_W
        p2_H = int(pixel_emb.shape[2])
        p2_W = int(pixel_emb.shape[3])

        # Reshape logits to spatial, then interpolate
        ml_spatial: Tensor = mask_logits.reshape(B, N, p2_H, p2_W)  # (B, N, H/4, W/4)
        # Merge B and N for interpolation
        ml_bn: Tensor = ml_spatial.reshape(B * N, 1, p2_H, p2_W)
        if p2_H != mem_H or p2_W != mem_W:
            ml_bn = F.interpolate(
                ml_bn, size=(mem_H, mem_W), mode="bilinear", align_corners=False
            )
        ml_resized: Tensor = ml_bn.reshape(B, N, mem_H * mem_W)  # (B, N, S_fpn)

        # Convert to attention mask: where sigmoid < 0.5 → block (True)
        sig: Tensor = F.sigmoid(ml_resized)  # (B, N, S_fpn)

        # Build (N, S_fpn) mask by averaging over batch (conservative: block if
        # majority of batch items want to block)
        # For simplicity, use the first batch item or average
        mask_data: list[list[float]] = []
        for n in range(N):
            row_m: list[float] = []
            for s in range(S_fpn):
                avg = 0.0
                for b in range(B):
                    avg += float(sig[b, n, s].item())
                avg /= B
                # Block (large negative) where average sigmoid < 0.5
                row_m.append(0.0 if avg >= 0.5 else -1e4)
            mask_data.append(row_m)

        return lucid.tensor(mask_data, device=queries.device)  # (N, S_fpn)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: dict[str, Tensor] | None = None,
    ) -> SemanticSegmentationOutput:
        """Run Mask2Former.

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

        # 1. Backbone
        features: list[Tensor] = self.backbone.forward(x)

        # 2. Multi-scale pixel decoder → [P3, P4, P5] + pixel_emb
        fpn_levels, pixel_emb = self.pixel_decoder.forward(features)
        pixel_emb = cast(Tensor, self.pixel_proj(pixel_emb))

        fH = int(pixel_emb.shape[2])
        fW = int(pixel_emb.shape[3])
        d = int(pixel_emb.shape[1])

        # 3. Initialise queries: (N, B, d)
        N = self._cfg.num_queries
        q_embed: Tensor = cast(Tensor, self.query_embed.weight)  # (N, d)
        tgt: Tensor = q_embed.unsqueeze(1).expand(-1, B, -1)  # (N, B, d)

        # 4. Masked-attention decoder layers
        num_levels = self._cfg.num_feature_levels
        for i, dec_layer in enumerate(self.decoder_layers):
            # Pick FPN level (cycle: 0=P3, 1=P4, 2=P5)
            level_idx = i % num_levels
            fpn_feat: Tensor = fpn_levels[level_idx]  # (B, fpn_ch, H_l, W_l)
            lH = int(fpn_feat.shape[2])
            lW = int(fpn_feat.shape[3])

            # Flatten to (S_l, B, fpn_ch)
            fpn_ch = int(fpn_feat.shape[1])
            mem: Tensor = fpn_feat.reshape(B, fpn_ch, lH * lW).permute(2, 0, 1)

            # Compute masked attention mask: (N, S_l)
            attn_mask: Tensor = self._compute_mask_attn(tgt, pixel_emb, lH, lW)

            tgt = dec_layer.forward(tgt, mem, attn_mask=attn_mask)

        # 5. Prediction heads
        hs_bn: Tensor = tgt.permute(1, 0, 2)  # (B, N, d)
        class_logits: Tensor = cast(Tensor, self.class_head(hs_bn))  # (B, N, K+1)

        mask_emb_out: Tensor = cast(Tensor, self.mask_embed(hs_bn))  # (B, N, d)
        pixel_flat: Tensor = pixel_emb.reshape(B, d, fH * fW)  # (B, d, S)
        mask_logits_flat: Tensor = lucid.matmul(mask_emb_out, pixel_flat)
        mask_logits: Tensor = mask_logits_flat.reshape(B, N, fH, fW)

        # 6. Seg logits (inference)
        K_plus_1 = self._cfg.num_classes + 1
        class_probs: Tensor = F.softmax(class_logits, dim=-1)  # (B, N, K+1)
        mask_probs: Tensor = F.sigmoid(mask_logits)  # (B, N, H/4, W/4)

        class_probs_t: Tensor = class_probs.permute(0, 2, 1)  # (B, K+1, N)
        mask_flat: Tensor = mask_probs.reshape(B, N, fH * fW)  # (B, N, S)
        seg_flat: Tensor = lucid.matmul(class_probs_t, mask_flat)
        seg_logits_small: Tensor = seg_flat.reshape(B, K_plus_1, fH, fW)

        seg_logits: Tensor = F.interpolate(
            seg_logits_small, size=(iH, iW), mode="bilinear", align_corners=False
        )

        # 7. Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(class_logits, mask_logits, targets, (fH, fW))

        return SemanticSegmentationOutput(logits=seg_logits, loss=loss)

    def _compute_loss(
        self,
        class_logits: Tensor,
        mask_logits: Tensor,
        targets: dict[str, Tensor],
        feat_size: tuple[int, int],
    ) -> Tensor:
        """Hungarian-matched mask + class loss across batch."""
        B = int(class_logits.shape[0])
        N = int(class_logits.shape[1])
        K = self._cfg.num_classes
        fH, fW = feat_size

        gt_masks_full: Tensor = targets["masks"]

        cls_losses: list[Tensor] = []
        mask_losses: list[Tensor] = []

        for b in range(B):
            cl_b: Tensor = class_logits[b]
            ml_b: Tensor = mask_logits[b]
            seg_b: Tensor = gt_masks_full[b]

            seg_H = int(seg_b.shape[0])
            seg_W = int(seg_b.shape[1])
            seen: set[int] = set()
            unique_classes: list[int] = []
            for h in range(seg_H):
                for w in range(seg_W):
                    c = int(seg_b[h, w].item())
                    if c not in seen:
                        seen.add(c)
                        unique_classes.append(c)

            M = len(unique_classes)

            if M == 0:
                bg_tgt_data: list[int] = [K] * N
                bg_tgt: Tensor = lucid.tensor(bg_tgt_data)
                cls_losses.append(F.cross_entropy(cl_b, bg_tgt, reduction="mean"))
                continue

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

            gt_labels_t: Tensor = lucid.tensor(gt_label_data)
            gt_masks_t: Tensor = lucid.tensor(gt_masks_list)

            gt_masks_small: Tensor = F.interpolate(
                gt_masks_t.reshape(M, 1, seg_H, seg_W),
                size=(fH, fW),
                mode="nearest",
            ).reshape(M, fH, fW)

            pred_idx, gt_idx = _hungarian_match_masks(
                cl_b, ml_b, gt_labels_t, gt_masks_small
            )

            cls_tgt_data: list[int] = [K] * N
            for pi, gi in zip(pred_idx, gt_idx):
                cls_tgt_data[pi] = int(gt_labels_t[gi].item())
            cls_tgt: Tensor = lucid.tensor(cls_tgt_data)
            cls_losses.append(F.cross_entropy(cl_b, cls_tgt, reduction="mean"))

            if pred_idx:
                for pi, gi in zip(pred_idx, gt_idx):
                    pred_m: Tensor = ml_b[pi]
                    gt_m: Tensor = gt_masks_small[gi]
                    mask_losses.append(
                        F.binary_cross_entropy_with_logits(
                            pred_m.reshape(-1), gt_m.reshape(-1)
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
