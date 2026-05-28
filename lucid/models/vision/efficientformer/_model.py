"""EfficientFormer backbone and classifier (Li et al., 2022).

Paper: "EfficientFormer: Vision Transformers at MobileNet Speed"

Key ideas:
  1. MetaFormer insight: most of the capacity comes from the MLP, not the
     token mixer. Replace global attention with cheap local pooling in early
     stages.
  2. All-4D processing in early stages (B,C,H,W) — no reshape overhead.
  3. Stage 4 (last stage only): switch to standard MHA for global context.
  4. Depthwise conv stem (2×stride-2) for efficient spatial downsampling.

Architecture (EfficientFormer-L1, image=224):
  Stem   : Conv3×3(s=2) → Conv3×3(s=2) → (56×56, 48)
  Stage 1: 3 × PoolBlock(48)            → Downsample → (28×28, 96)
  Stage 2: 2 × PoolBlock(96)            → Downsample → (14×14, 224)
  Stage 3: 6 × PoolBlock(224)           → Downsample → (7×7,   448)
  Stage 4: 4 × AttnBlock(448)
  Head   : mean pool spatial → LN → FC
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._classification import DropPath, LayerScale
from lucid.models.vision.efficientformer._config import EfficientFormerConfig

# ---------------------------------------------------------------------------
# Pooling token mixer (MetaFormer-style)
# ---------------------------------------------------------------------------


class _PoolingBlock(nn.Module):
    """AvgPool3×3 − identity (pool the context, subtract self to get context diff)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.pool(x)) - x


# ---------------------------------------------------------------------------
# MLP (channel-last friendly)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.fc2(F.gelu(cast(Tensor, self.fc1(x)))))


# ---------------------------------------------------------------------------
# Stage 1-3 block: pooling-based (4D spatial tensors)
# ---------------------------------------------------------------------------


class _EfficientFormerPoolBlock(nn.Module):
    """Pooling MetaFormer block operating in (B, C, H, W) layout."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        drop_path_rate: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.pool_mixer = _PoolingBlock()
        self.norm = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # DWConv + BN (LePE-style local position encoding, no LayerScale).
        shortcut = x
        x = cast(Tensor, self.bn(cast(Tensor, self.dwconv(x))))
        x = shortcut + x

        # Pooling mixer (spatial) with LayerScale + DropPath.
        x = x + cast(
            Tensor,
            self.drop_path(cast(Tensor, self.ls1(cast(Tensor, self.pool_mixer(x))))),
        )

        # MLP (channel-last) with LayerScale + DropPath. LayerScale applied
        # in (B, C, H, W) layout after permute-back.
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1)
        x_cl = cast(Tensor, self.norm(x_cl))
        x_cl = cast(Tensor, self.mlp(x_cl))
        x_mlp = x_cl.permute(0, 3, 1, 2)
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls2(x_mlp))))
        return x


# ---------------------------------------------------------------------------
# Stage 4 block: attention-based (sequence layout)
# ---------------------------------------------------------------------------


class _EfficientFormerAttnBlock(nn.Module):
    """Standard MHA transformer block for stage 4."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path_rate: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C)
        n = cast(Tensor, self.norm1(x))
        attn_out, _ = self.attn(n, n, n)
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls1(attn_out))))
        m = cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls2(m))))
        return x


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.bn(cast(Tensor, self.conv(x))))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_efficientformer(cfg: EfficientFormerConfig) -> tuple[
    nn.Sequential,  # stem
    nn.ModuleList,  # stages (pool stages + attn stage)
    nn.ModuleList,  # downsamplers (len = num_stages - 1)
    nn.LayerNorm,  # head norm
    list[FeatureInfo],
    int,
]:
    # Stem: 2× Conv3×3 stride=2
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, cfg.embed_dims[0] // 2, 3, stride=2, padding=1),
        nn.BatchNorm2d(cfg.embed_dims[0] // 2),
        nn.ReLU(),
        nn.Conv2d(cfg.embed_dims[0] // 2, cfg.embed_dims[0], 3, stride=2, padding=1),
        nn.BatchNorm2d(cfg.embed_dims[0]),
        nn.ReLU(),
    )

    num_stages = len(cfg.depths)
    last_stage = num_stages - 1

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4  # stem applies 4× downsampling

    # Linear DropPath schedule across the whole trunk (paper §4.1).
    total_blocks = sum(cfg.depths)
    if total_blocks > 1 and cfg.drop_path_rate > 0.0:
        dp_rates = [
            cfg.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)
        ]
    else:
        dp_rates = [cfg.drop_path_rate] * total_blocks
    cursor = 0

    for i, (depth, dim, mlp_ratio) in enumerate(
        zip(cfg.depths, cfg.embed_dims, cfg.mlp_ratios)
    ):
        if i < last_stage:
            # Pooling-based blocks (4D)
            stage = nn.Sequential(
                *[
                    _EfficientFormerPoolBlock(
                        dim, mlp_ratio, dp_rates[cursor + j], cfg.layer_scale_init
                    )
                    for j in range(depth)
                ]
            )
        else:
            # Attention-based blocks (sequence) — num_heads auto
            num_heads = max(1, dim // 32)
            stage = nn.Sequential(
                *[
                    _EfficientFormerAttnBlock(
                        dim,
                        num_heads,
                        mlp_ratio,
                        dp_rates[cursor + j],
                        cfg.layer_scale_init,
                    )
                    for j in range(depth)
                ]
            )
        cursor += depth
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < num_stages - 1:
            downsamplers.append(_Downsample(dim, cfg.embed_dims[i + 1]))
            reduction *= 2

    head_norm = nn.LayerNorm(cfg.embed_dims[-1])
    return (
        stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.embed_dims[-1],
    )


# ---------------------------------------------------------------------------
# EfficientFormer backbone
# ---------------------------------------------------------------------------


class EfficientFormer(PretrainedModel, BackboneMixin):
    r"""EfficientFormer backbone (Li et al., 2022).

    EfficientFormer is a *mobile-grade* vision backbone designed so
    that its on-device latency, rather than its FLOP count, matches
    MobileNet on the same hardware while retaining transformer-level
    accuracy.  The trunk has four stages preceded by a 2x stride-2
    convolutional stem.  Stages 1-3 are MetaFormer-style *pooling*
    blocks operating in :math:`(B, C, H, W)` layout — no reshape, no
    self-attention:

    .. math::

        \begin{aligned}
        x &\leftarrow x + \gamma_1 \odot \bigl(
            \mathrm{AvgPool}_{3 \times 3}(x) - x\bigr), \\
        x &\leftarrow x + \gamma_2 \odot \mathrm{MLP}(x),
        \end{aligned}

    with :math:`\gamma_1, \gamma_2` initialised to :math:`10^{-5}`
    (CaiT-style layer scale).  Stage 4 is the only stage that pays the
    cost of a reshape and runs standard multi-head self-attention on
    the now-tiny token grid.

    :meth:`forward_features` returns the mean-pooled
    :math:`(B, \texttt{embed\_dims[-1]})` feature.

    Parameters
    ----------
    config : EfficientFormerConfig
        Frozen dataclass specifying ``depths``, ``embed_dims``,
        ``mlp_ratios``, ``drop_path_rate``, ``layer_scale_init``,
        ``in_channels``, and ``num_classes``.  See
        :class:`EfficientFormerConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Two stride-2 :math:`3 \times 3` convolutions reducing the
        input by 4x.
    stages : nn.ModuleList
        Four stages — three pooling stages and one attention stage.
    downsamplers : nn.ModuleList
        Three between-stage stride-2 convolutional downsamplers.
    head_norm : nn.LayerNorm
        Final LayerNorm applied to the channel-last pooled feature map.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference: Yanyu Li *et al.*, *"EfficientFormer: Vision
    Transformers at MobileNet Speed"*, NeurIPS 2022,
    `arXiv:2206.01191 <https://arxiv.org/abs/2206.01191>`_.

    Examples
    --------
    Build an EfficientFormer-L1 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.efficientformer import (
    ...     EfficientFormer, EfficientFormerConfig,
    ... )
    >>> model = EfficientFormer(EfficientFormerConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, embed_dims[-1])
    (1, 448)
    """

    config_class: ClassVar[type[EfficientFormerConfig]] = EfficientFormerConfig
    base_model_prefix: ClassVar[str] = "efficientformer"

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, fi, out_dim = _build_efficientformer(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._feature_info = fi
        self._out_dim = out_dim

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        num_stages = len(self.stages)
        last_stage = num_stages - 1

        for i, stage in enumerate(self.stages):
            if i < last_stage:
                # Pool stages: (B, C, H, W)
                x = cast(Tensor, stage(x))
            else:
                # Attention stage: flatten spatial → (B, N, C) → run → reshape back
                B, C, H, W = x.shape
                x_seq = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
                x_seq = cast(Tensor, stage(x_seq))
                x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)

            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))

        # Global mean pool → (B, C)
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        x_cl = cast(Tensor, self.head_norm(x_cl))
        return x_cl.mean(dim=1)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# EfficientFormer for image classification
# ---------------------------------------------------------------------------


class EfficientFormerForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""EfficientFormer with a linear classification head (Li et al., 2022).

    Wraps the same trunk as :class:`EfficientFormer` (stem + three
    pooling stages + one attention stage) and adds a mean pool over
    tokens, a final LayerNorm, and a single :class:`nn.Linear`
    classification head:

    .. math::

        \text{logits} = W_{\text{head}}\,
            \mathrm{Mean}(\mathrm{LN}(z^{L}))
            + b_{\text{head}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : EfficientFormerConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See
        :class:`EfficientFormerConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Two stride-2 convolutions.
    stages : nn.ModuleList
        Three pooling stages + one attention stage.
    downsamplers : nn.ModuleList
        Three between-stage downsamplers.
    head_norm : nn.LayerNorm
        Final LayerNorm.
    classifier : nn.Linear
        Final linear projection of width
        ``(num_classes, embed_dims[-1])``.

    Notes
    -----
    Reference: Li *et al.*, *"EfficientFormer: Vision Transformers at
    MobileNet Speed"*, NeurIPS 2022.  EfficientFormer-L1 / L3 / L7
    reach **79.2 / 82.4 / 83.3 % top-1 on ImageNet-1k** at MobileNetV2-
    class on-device latency (Li et al., 2022, Table 4).

    Examples
    --------
    End-to-end inference with the default EfficientFormer-L1
    classifier:

    >>> import lucid
    >>> from lucid.models.vision.efficientformer import (
    ...     EfficientFormerConfig, EfficientFormerForImageClassification,
    ... )
    >>> model = EfficientFormerForImageClassification(EfficientFormerConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[EfficientFormerConfig]] = EfficientFormerConfig
    base_model_prefix: ClassVar[str] = "efficientformer"

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, _, out_dim = _build_efficientformer(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        num_stages = len(self.stages)
        last_stage = num_stages - 1

        for i, stage in enumerate(self.stages):
            if i < last_stage:
                x = cast(Tensor, stage(x))
            else:
                B, C, H, W = x.shape
                x_seq = x.flatten(2).permute(0, 2, 1)
                x_seq = cast(Tensor, stage(x_seq))
                x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)

            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))

        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_cl = cast(Tensor, self.head_norm(x_cl))
        feat = x_cl.mean(dim=1)
        logits = cast(Tensor, self.classifier(feat))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
