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
    ↓  ResNet (conv1 → pool → layer1–layer4) — C5 (2048ch, stride 32)
    ↓  1×1 Conv: 2048 → d_model  (default 256)
    ↓  Flatten: (B, d_model, H', W') → (H'·W', B, d_model)
    ↓  + 2D sinusoidal positional encoding re-injected at every layer
  Encoder: N_enc × (self-attention + FFN)
  Decoder: N_dec × (self-attention on queries + cross-attention to memory + FFN)
    ↓  (N_queries, B, d_model)
  Class head: Linear → (B, N, num_classes + 1)   softmax at inference
  Box head:   3-layer MLP → (B, N, 4)   sigmoid to enforce [0,1]

Faithfulness notes
------------------
* ResNet C5 feature map (stride 32) with **frozen** batch-norm (eval-only
  affine + running-stat math, no ``num_batches_tracked``).
* d_model=256, n_head=8, 6 enc / 6 dec layers, dim_ffn=2048.
* N=100 object queries.
* Sinusoidal 2-D positional encoding identical to the reference
  ``PositionEmbeddingSine`` (num_pos_feats=d_model/2, temperature=10000,
  normalize=True, scale=2π) — verified to ~1e-6.
* The transformer (``_DETRTransformer``) mirrors the reference layer
  layout verbatim: ``encoder.layers.{N}`` (no final encoder norm),
  ``decoder.layers.{N}`` + a final ``decoder.norm`` LayerNorm.  Positional
  encodings are re-injected at every layer (added to Q/K of encoder
  self-attention; query positions added to Q/K of decoder self-attention
  and to Q of decoder cross-attention, spatial positions added to the
  memory K), exactly as the reference forward.  Post-norm
  (normalize_before=False), ReLU activation.
* Hungarian cost matrix: L_cls + 5·L_L1 + 2·L_GIoU (matched queries).
* Background class weight 0.1 for unmatched queries.
"""

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
    solve_assignment,
)
from lucid.models.vision.detr._config import DETRConfig

# ---------------------------------------------------------------------------
# Frozen BatchNorm (eval-only affine + running-stat math, no nbt buffer)
# ---------------------------------------------------------------------------


class _FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen affine params + running stats.

    Holds exactly four persistent buffers — ``weight``, ``bias``,
    ``running_mean``, ``running_var`` — with **no** ``num_batches_tracked``,
    matching the reference DETR ``FrozenBatchNorm2d`` key-set.  The forward
    applies the eval-time normalisation

    .. math::

        y = (x - \\mathrm{running\\_mean})
            \\cdot \\mathrm{rsqrt}(\\mathrm{running\\_var} + \\varepsilon)
            \\cdot \\mathrm{weight} + \\mathrm{bias}

    with :math:`\\varepsilon = 10^{-5}`, regardless of ``train`` / ``eval``
    mode (the statistics never update).
    """

    eps: ClassVar[float] = 1e-5

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.register_buffer("weight", lucid.ones(num_features))
        self.register_buffer("bias", lucid.zeros(num_features))
        self.register_buffer("running_mean", lucid.zeros(num_features))
        self.register_buffer("running_var", lucid.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        w = cast(Tensor, self.weight).reshape(1, -1, 1, 1)
        b = cast(Tensor, self.bias).reshape(1, -1, 1, 1)
        rm = cast(Tensor, self.running_mean).reshape(1, -1, 1, 1)
        rv = cast(Tensor, self.running_var).reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# ---------------------------------------------------------------------------
# ResNet backbone (C5 only — frozen BN, reference key layout)
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
        self.bn1 = _FrozenBatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = _FrozenBatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = _FrozenBatchNorm2d(out_ch)
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
            _FrozenBatchNorm2d(out_ch),
        )
    blocks: list[nn.Module] = [_Bottleneck(in_ch, mid_ch, stride=stride, downsample=ds)]
    for _ in range(1, num_blocks):
        blocks.append(_Bottleneck(out_ch, mid_ch))
    return nn.Sequential(*blocks), out_ch


class _ResNetC5(nn.Module):
    """ResNet backbone, returns C5 feature map (stride 32, 2048ch).

    Submodule names (``conv1`` / ``bn1`` / ``layer1``…``layer4``) mirror
    the reference ResNet body so the converter map for the backbone is a
    pure prefix strip (``backbone.0.body.<rest>`` → ``backbone.<rest>``).
    """

    def __init__(self, in_channels: int, layers: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = _FrozenBatchNorm2d(64)
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
# DETR-exact 2-D sinusoidal positional encoding (PositionEmbeddingSine)
# ---------------------------------------------------------------------------


class _PositionEmbeddingSine(nn.Module):
    """Reference ``PositionEmbeddingSine`` for an unpadded feature map.

    Builds the sinusoidal 2-D positional encoding the reference DETR feeds
    into every attention layer.  Inference always runs on a single,
    unpadded image, so the "not-mask" is all-ones and reduces to plain
    row / column cumulative sums.

    Parameters mirror the reference defaults: ``num_pos_feats = d_model/2``,
    ``temperature = 10000``, ``normalize = True``, ``scale = 2π``.
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: float = 10000.0,
        scale: float = 6.283185307179586,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale
        self.eps = eps

    def forward(self, batch: int, height: int, width: int, device: str) -> Tensor:  # type: ignore[override]
        npf = self.num_pos_feats
        ones = lucid.ones(1, height, width, dtype=lucid.float32, device=device)
        y_embed = ones.cumsum(dim=1)
        x_embed = ones.cumsum(dim=2)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = lucid.arange(0, npf, dtype=lucid.float32, device=device)
        dim_t = self.temperature ** (2.0 * lucid.floor(dim_t / 2.0) / npf)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = lucid.stack(
            [lucid.sin(pos_x[:, :, :, 0::2]), lucid.cos(pos_x[:, :, :, 1::2])], dim=4
        ).flatten(3)
        pos_y = lucid.stack(
            [lucid.sin(pos_y[:, :, :, 0::2]), lucid.cos(pos_y[:, :, :, 1::2])], dim=4
        ).flatten(3)
        pos = lucid.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (1, d, H, W)
        if batch > 1:
            pos = pos.repeat(batch, 1, 1, 1)
        return pos


# ---------------------------------------------------------------------------
# DETR transformer (reference layer layout + per-layer pos re-injection)
# ---------------------------------------------------------------------------


def _with_pos(tensor: Tensor, pos: Tensor | None) -> Tensor:
    """Add positional embedding to a tensor (identity when ``pos`` is None)."""
    return tensor if pos is None else tensor + pos


class _DETREncoderLayer(nn.Module):
    """One post-norm encoder layer with pos re-injection on Q/K.

    Submodule names match the reference: ``self_attn`` (fused-``in_proj``
    :class:`nn.MultiheadAttention`), ``linear1`` / ``linear2``, ``norm1`` /
    ``norm2``.  Dropout is a no-op in eval mode.
    """

    def __init__(
        self, d_model: int, n_head: int, dim_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, pos: Tensor | None) -> Tensor:  # type: ignore[override]
        q = _with_pos(src, pos)
        k = q
        src2 = self.self_attn(q, k, value=src, need_weights=False)[0]
        src = cast(Tensor, self.norm1(src + cast(Tensor, self.dropout1(src2))))
        src2 = cast(
            Tensor,
            self.linear2(
                cast(Tensor, self.dropout(F.relu(cast(Tensor, self.linear1(src)))))
            ),
        )
        src = cast(Tensor, self.norm2(src + cast(Tensor, self.dropout2(src2))))
        return src


class _DETRDecoderLayer(nn.Module):
    """One post-norm decoder layer.

    Submodule names match the reference: ``self_attn``, ``multihead_attn``
    (both fused-``in_proj`` :class:`nn.MultiheadAttention`), ``linear1`` /
    ``linear2``, ``norm1`` / ``norm2`` / ``norm3``.
    """

    def __init__(
        self, d_model: int, n_head: int, dim_ff: int, dropout: float
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(  # type: ignore[override]
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor | None,
        query_pos: Tensor | None,
    ) -> Tensor:
        q = _with_pos(tgt, query_pos)
        k = q
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = cast(Tensor, self.norm1(tgt + cast(Tensor, self.dropout1(tgt2))))
        tgt2 = self.multihead_attn(
            _with_pos(tgt, query_pos),
            _with_pos(memory, pos),
            value=memory,
            need_weights=False,
        )[0]
        tgt = cast(Tensor, self.norm2(tgt + cast(Tensor, self.dropout2(tgt2))))
        tgt2 = cast(
            Tensor,
            self.linear2(
                cast(Tensor, self.dropout(F.relu(cast(Tensor, self.linear1(tgt)))))
            ),
        )
        tgt = cast(Tensor, self.norm3(tgt + cast(Tensor, self.dropout3(tgt2))))
        return tgt


class _DETREncoder(nn.Module):
    """Stack of encoder layers — NO final norm (reference removes it)."""

    def __init__(
        self, d_model: int, n_head: int, dim_ff: int, dropout: float, depth: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DETREncoderLayer(d_model, n_head, dim_ff, dropout) for _ in range(depth)]
        )

    def forward(self, src: Tensor, pos: Tensor | None) -> Tensor:  # type: ignore[override]
        output = src
        for layer in self.layers:
            output = cast(Tensor, layer(output, pos=pos))
        return output


class _DETRDecoder(nn.Module):
    """Stack of decoder layers + a final ``norm`` (LayerNorm).

    For inference only the last (norm-applied) decoder output matters, so
    this returns the single final activation (no intermediate stack).
    """

    def __init__(
        self, d_model: int, n_head: int, dim_ff: int, dropout: float, depth: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_DETRDecoderLayer(d_model, n_head, dim_ff, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(  # type: ignore[override]
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Tensor | None,
        query_pos: Tensor | None,
    ) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = cast(Tensor, layer(output, memory, pos=pos, query_pos=query_pos))
        return cast(Tensor, self.norm(output))


class _DETRTransformer(nn.Module):
    """Reference DETR transformer: ``encoder`` + ``decoder``.

    Forward takes the projected feature map ``(B, d, H, W)``, the spatial
    positional encoding ``pos`` ``(B, d, H, W)`` and the learned query
    embedding ``query_embed`` ``(N, d)``; it flattens, runs the encoder /
    decoder with per-layer positional re-injection, and returns the final
    decoder activation as ``(B, N, d)``.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = _DETREncoder(
            d_model, n_head, dim_feedforward, dropout, num_encoder_layers
        )
        self.decoder = _DETRDecoder(
            d_model, n_head, dim_feedforward, dropout, num_decoder_layers
        )

    def forward(  # type: ignore[override]
        self, src: Tensor, pos_embed: Tensor, query_embed: Tensor
    ) -> Tensor:
        B = int(src.shape[0])
        c = int(src.shape[1])
        h = int(src.shape[2])
        w = int(src.shape[3])

        # (B, c, H, W) → (H*W, B, c)
        src_flat = src.reshape(B, c, h * w).permute(2, 0, 1)
        pos_flat = pos_embed.reshape(B, c, h * w).permute(2, 0, 1)
        # (N, c) → (N, B, c)
        query_pos = query_embed.unsqueeze(1).repeat(1, B, 1)
        tgt = lucid.zeros(query_pos.shape, device=src.device.type)

        memory = cast(Tensor, self.encoder(src_flat, pos_flat))
        hs = cast(
            Tensor, self.decoder(tgt, memory, pos=pos_flat, query_pos=query_pos)
        )  # (N, B, c)
        return hs.permute(1, 0, 2)  # (B, N, c)


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

    # Build M × N cost matrix (rows = GTs, cols = queries — M ≤ N for DETR)
    cost: list[list[float]] = []
    for m in range(M):
        gt_cls = int(gt_labels[m].item())
        row: list[float] = []
        for n in range(N):
            c_cls = -float(scores[n, gt_cls].item())
            c_l1 = sum(
                abs(float(pred_boxes[n, d].item()) - float(gt_boxes[m, d].item()))
                for d in range(4)
            )
            c_giou = -float(giou_mat[n, m].item())
            row.append(cost_cls * c_cls + cost_l1 * c_l1 + cost_giou * c_giou)
        cost.append(row)

    gt_idx, pred_idx = solve_assignment(cost)
    return pred_idx, gt_idx


# ---------------------------------------------------------------------------
# DETR model
# ---------------------------------------------------------------------------


class DETRForObjectDetection(PretrainedModel):
    r"""DETR end-to-end object detector (Carion et al., ECCV 2020).

    The first detection model to formulate object detection as a direct
    **set prediction** problem, eliminating anchors and NMS.  A ResNet
    backbone produces a stride-32 feature map :math:`(B, C, H', W')` that
    is projected to :math:`d_\mathrm{model}` channels, flattened to a
    sequence, and processed by a stack of transformer encoder layers.
    A learned set of :math:`N` *object queries* (default 100) is then
    decoded against this memory by the transformer decoder, with each
    query producing one class logit vector (K+1, including
    "no-object") and one box prediction :math:`(c_x, c_y, w, h)`
    normalised to :math:`[0, 1]`.

    During training, predictions and ground-truth boxes are matched by
    the Hungarian algorithm under a cost combining class probability and
    box L1 + GIoU, and the same combination defines the per-pair loss.

    Parameters
    ----------
    config : DETRConfig
        Frozen architecture spec.  Use :func:`detr_resnet50` /
        :func:`detr_resnet101` for the paper-cited variants.

    Attributes
    ----------
    config : DETRConfig
        Stored copy of the config that built this model.
    backbone : _ResNetC5
        ResNet trunk through stage 5 (frozen BN), producing a stride-32
        feature map.
    input_proj : nn.Conv2d
        1x1 conv projecting backbone output to ``d_model``.
    query_embed : nn.Embedding
        Learned object queries of shape ``(num_queries, d_model)``.
    transformer : _DETRTransformer
        Reference DETR transformer (``encoder.layers.{N}`` with no final
        encoder norm; ``decoder.layers.{N}`` + a final ``decoder.norm``),
        with per-layer positional re-injection.
    class_embed : nn.Linear
        Per-query class head with output dim ``num_classes + 1`` (the
        extra "no-object" slot is essential for matching).
    bbox_embed : _MLP
        Three-layer MLP producing per-query 4-D box predictions; the
        final activation is a sigmoid so boxes lie in :math:`[0, 1]`.

    Notes
    -----
    See Carion et al., "End-to-End Object Detection with Transformers",
    ECCV 2020 (arXiv:2005.12872).  The bipartite Hungarian assignment
    finds the permutation :math:`\hat{\sigma} \in \mathfrak{S}_N` that
    minimises

    .. math::

        \hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N}
                       \sum_{i=1}^{N}
                       \mathcal{L}_\mathrm{match}\bigl(y_i, \hat{y}_{\sigma(i)}\bigr),

    with the pair cost

    .. math::

        \mathcal{L}_\mathrm{match}(y, \hat{y}) =
            -\mathbb{1}_{c \ne \varnothing}\,\hat{p}(c)
            + \mathbb{1}_{c \ne \varnothing}\,
              \bigl(\lambda_{\mathrm{L_1}}\|b - \hat{b}\|_1
                    + \lambda_\mathrm{GIoU}\,
                      (1 - \mathrm{GIoU}(b, \hat{b}))\bigr).

    The final loss reuses the same per-pair terms, with "no-object"
    matches penalised at reduced weight.  Removing NMS / anchors makes
    DETR conceptually simple but training-data hungry — the paper
    reports needing 500 epochs to fully converge on COCO.

    Examples
    --------
    Inference returns class logits and normalised cxcywh boxes for every
    query:

    >>> import lucid
    >>> from lucid.models.vision.detr import detr_resnet50
    >>> model = detr_resnet50()
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape   # (B, num_queries, K + 1)
    (1, 100, 81)
    >>> out.pred_boxes.shape
    (1, 100, 4)

    Training pass with normalised xyxy ground-truth boxes (Hungarian
    matching computes the set loss):

    >>> targets = [{
    ...     "boxes":  lucid.tensor([[0.1, 0.1, 0.5, 0.5]]),
    ...     "labels": lucid.tensor([3], dtype=lucid.int64),
    ... }]
    >>> out = model(x, targets=targets)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[DETRConfig]] = DETRConfig
    base_model_prefix: ClassVar[str] = "detr"

    def __init__(self, config: DETRConfig) -> None:
        super().__init__(config)
        self._cfg = config
        d = config.d_model

        # Backbone (frozen BN)
        self.backbone = _ResNetC5(config.in_channels, config.backbone_layers)
        self.input_proj = nn.Conv2d(self.backbone.out_channels, d, 1)

        # Object queries
        self.query_embed = nn.Embedding(config.num_queries, d)

        # Transformer
        self.transformer = _DETRTransformer(
            d_model=d,
            n_head=config.n_head,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Positional encoding (reference PositionEmbeddingSine)
        self._pos_embed = _PositionEmbeddingSine(num_pos_feats=d // 2)

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
        device = feat.device.type

        # 2. 2-D positional encoding (B, d, H', W')
        pos_embed = self._pos_embed.forward(B, fH, fW, device)

        # 3. Transformer — returns (B, N, d)
        queries: Tensor = cast(Tensor, self.query_embed.weight)  # (N, d)
        hs_bn = cast(Tensor, self.transformer(feat, pos_embed, queries))  # (B, N, d)

        # 4. Prediction heads
        logits: Tensor = cast(Tensor, self.class_embed(hs_bn))  # (B, N, K+1)
        pred_boxes: Tensor = F.sigmoid(
            cast(Tensor, self.bbox_embed(hs_bn))
        )  # (B, N, 4)

        # 5. Loss
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

            lucid.tensor(cls_targets_data)
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
