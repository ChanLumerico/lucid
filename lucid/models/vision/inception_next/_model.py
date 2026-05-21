"""InceptionNeXt backbone and classifier (Yu et al., 2023).

Paper: "InceptionNeXt: When Inception Meets ConvNeXt"

Key ideas:
  1. ConvNeXt's large 7×7 DWConv is decomposed into parallel branches
     (Inception-style) to reduce computation while maintaining receptive field.
  2. The token_mixer decomposes into three depthwise conv branches plus an
     identity passthrough (branch_ratio=0.125 channels per named branch).
  3. Block uses BatchNorm2d (not LayerNorm) and 1×1 Conv2d MLP (not Linear),
     operating entirely in NCHW space — matches timm's MetaNeXtBlock.
  4. Same patchify stem (Conv2d + BN2d), downsampling (BN + stride-2 Conv2d),
     and MlpClassifierHead (fc1 → GELU → LN → fc2) as timm's InceptionNeXt.

State-dict naming matches timm inception_next_tiny / small / base exactly:
  stem.0.*            Conv2d
  stem.1.*            BatchNorm2d
  stages.N.downsample.0.*   BatchNorm2d  (absent for stage 0 — Identity)
  stages.N.downsample.1.*   Conv2d
  stages.N.blocks.M.gamma
  stages.N.blocks.M.token_mixer.dwconv_hw.*   3×3 DWConv
  stages.N.blocks.M.token_mixer.dwconv_w.*    1×K DWConv
  stages.N.blocks.M.token_mixer.dwconv_h.*    K×1 DWConv
  stages.N.blocks.M.norm.*   BatchNorm2d
  stages.N.blocks.M.mlp.fc1.*   Conv2d 1×1
  stages.N.blocks.M.mlp.fc2.*   Conv2d 1×1
  head.fc1.*    Linear (expand by mlp_ratio=3)
  head.norm.*   LayerNorm
  head.fc2.*    Linear (→ num_classes)
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.inception_next._config import InceptionNeXtConfig

# ---------------------------------------------------------------------------
# InceptionDWConv2d — token_mixer (named to match timm)
# ---------------------------------------------------------------------------


class _InceptionDWConv2d(nn.Module):
    """Three-branch depthwise conv mixer operating on channel splits.

    timm branch_ratio=0.125: gc = int(dim * 0.125) per named branch.
    Branches:
      identity passthrough  : dim - 3*gc channels
      dwconv_hw (3×3)       : gc channels
      dwconv_w  (1×K)       : gc channels  (band width)
      dwconv_h  (K×1)       : gc channels  (band height)
    """

    def __init__(
        self, dim: int, band_kernel: int = 11, branch_ratio: float = 0.125
    ) -> None:
        super().__init__()
        gc = int(dim * branch_ratio)
        self.gc = gc
        self.identity_chs = dim - 3 * gc

        pad = band_kernel // 2
        # Named exactly as timm's InceptionDWConv2d
        self.dwconv_hw = nn.Conv2d(gc, gc, 3, padding=1, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, (1, band_kernel), padding=(0, pad), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, (band_kernel, 1), padding=(pad, 0), groups=gc)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        id_chs = self.identity_chs
        gc = self.gc
        x0 = x[:, :id_chs, :, :]
        x1 = x[:, id_chs : id_chs + gc, :, :]
        x2 = x[:, id_chs + gc : id_chs + 2 * gc, :, :]
        x3 = x[:, id_chs + 2 * gc :, :, :]

        y0 = x0
        y1 = cast(Tensor, self.dwconv_hw(x1))
        y2 = cast(Tensor, self.dwconv_w(x2))
        y3 = cast(Tensor, self.dwconv_h(x3))
        return lucid.cat([y0, y1, y2, y3], dim=1)


# ---------------------------------------------------------------------------
# ConvMlp — 1×1 Conv2d MLP matching timm's ConvMlp
# ---------------------------------------------------------------------------


class _ConvMlp(nn.Module):
    """1×1 Conv2d MLP: fc1 → GELU → fc2 (NCHW, no norm inside)."""

    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = F.gelu(cast(Tensor, self.fc1(x)))
        return cast(Tensor, self.fc2(x))


# ---------------------------------------------------------------------------
# MetaNeXtBlock — ConvNeXt-style block with InceptionDWConv2d token_mixer
# ---------------------------------------------------------------------------


class _MetaNeXtBlock(nn.Module):
    """timm MetaNeXtBlock: token_mixer (NCHW) → BN → ConvMlp → LayerScale."""

    def __init__(
        self,
        dim: int,
        band_kernel: int,
        mlp_ratio: int,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.token_mixer = _InceptionDWConv2d(dim, band_kernel)
        self.norm = nn.BatchNorm2d(dim)
        self.mlp = _ConvMlp(dim, mlp_ratio)
        self.gamma = nn.Parameter(lucid.full((dim,), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x
        x = cast(Tensor, self.token_mixer(x))
        x = cast(Tensor, self.norm(x))
        x = cast(Tensor, self.mlp(x))
        x = x * self.gamma.reshape(-1, 1, 1)
        return shortcut + x


# ---------------------------------------------------------------------------
# Stage — holds downsample + blocks (matches timm's MetaNeXtStage)
# ---------------------------------------------------------------------------


class _Stage(nn.Module):
    """One InceptionNeXt stage: optional downsample + sequential blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        band_kernel: int,
        mlp_ratio: int,
        *,
        downsample: bool,
    ) -> None:
        super().__init__()
        if downsample:
            # timm: Sequential(BN2d(in_dim), Conv2d(in_dim→out_dim, 2, stride=2))
            self.downsample: nn.Module = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, 2, stride=2),
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[_MetaNeXtBlock(out_dim, band_kernel, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.downsample(x))
        return cast(Tensor, self.blocks(x))


# ---------------------------------------------------------------------------
# MlpClassifierHead — head.fc1 / head.norm / head.fc2 (matches timm)
# ---------------------------------------------------------------------------


class _MlpClassifierHead(nn.Module):
    """timm MlpClassifierHead: GlobalAvgPool → fc1 → GELU → norm → fc2."""

    def __init__(self, in_features: int, num_classes: int, mlp_ratio: int = 3) -> None:
        super().__init__()
        hidden = in_features * mlp_ratio
        self.fc1 = nn.Linear(in_features, hidden)
        # timm uses eps=1e-6 for the head LayerNorm
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W) — apply global avg pool, then mlp
        x = x.mean(dim=(2, 3))  # (B, C)
        x = cast(Tensor, self.fc1(x))
        x = F.gelu(x)
        x = cast(Tensor, self.norm(x))
        return cast(Tensor, self.fc2(x))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_inception_next(cfg: InceptionNeXtConfig) -> tuple[
    nn.Sequential,
    nn.ModuleList,
    list[FeatureInfo],
    int,
]:
    # stem.0 = Conv2d, stem.1 = BN2d
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, cfg.dims[0], 4, stride=4),
        nn.BatchNorm2d(cfg.dims[0]),
    )

    stages: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4

    mlp_ratios = cfg.mlp_ratios
    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        in_dim = cfg.dims[i - 1] if i > 0 else dim
        stage = _Stage(
            in_dim=in_dim,
            out_dim=dim,
            depth=depth,
            band_kernel=cfg.band_kernel,
            mlp_ratio=mlp_ratios[i],
            downsample=(i > 0),
        )
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        if i > 0:
            reduction *= 2

    return stem, nn.ModuleList(stages), fi, cfg.dims[-1]


# ---------------------------------------------------------------------------
# InceptionNeXt backbone
# ---------------------------------------------------------------------------


class InceptionNeXt(PretrainedModel, BackboneMixin):
    r"""InceptionNeXt backbone (Yu et al., 2024).

    A drop-in replacement for the ConvNeXt backbone that factorizes
    the single large :math:`7 \times 7` depthwise conv into four
    parallel Inception-style branches operating on disjoint channel
    splits:

    .. math::

        \mathrm{IDWConv}(x) = \mathrm{Concat}\bigl(
            x^{(1)},\;
            \mathrm{DW}_{3 \times 3}(x^{(2)}),\;
            \mathrm{DW}_{1 \times K}(x^{(3)}),\;
            \mathrm{DW}_{K \times 1}(x^{(4)})\bigr),

    with :math:`K = \texttt{band\_kernel}` (default 11).  The four
    branches together preserve a large effective receptive field
    while cutting depthwise FLOPs and memory traffic relative to a
    single :math:`7 \times 7` depthwise conv.  Each MetaNeXtBlock
    further applies BatchNorm, a 1x1 Conv-MLP, and a layer-scale
    parameter on the residual branch.

    :meth:`forward_features` returns the global-average-pooled
    :math:`(B, \texttt{dims[-1]})` feature.

    Parameters
    ----------
    config : InceptionNeXtConfig
        Frozen dataclass specifying ``depths``, ``dims``,
        ``band_kernel``, ``mlp_ratios``, ``in_channels``, and
        ``num_classes``.  See :class:`InceptionNeXtConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Patchify stem: :math:`4 \times 4` stride-4 Conv2d + BatchNorm2d.
    stages : nn.ModuleList
        Four :class:`_Stage` modules, each containing an optional
        downsampler and a sequence of MetaNeXt blocks.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference: Weihao Yu *et al.*, *"InceptionNeXt: When Inception
    Meets ConvNeXt"*, CVPR 2024,
    `arXiv:2303.16900 <https://arxiv.org/abs/2303.16900>`_.

    Examples
    --------
    Build an InceptionNeXt-T backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.inception_next import (
    ...     InceptionNeXt, InceptionNeXtConfig,
    ... )
    >>> model = InceptionNeXt(InceptionNeXtConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1])
    (1, 768)
    """

    config_class: ClassVar[type[InceptionNeXtConfig]] = InceptionNeXtConfig
    base_model_prefix: ClassVar[str] = "inception_next"

    def __init__(self, config: InceptionNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, fi, out_dim = _build_inception_next(config)
        self.stem = stem
        self.stages = stages
        self._feature_info = fi
        self._out_dim = out_dim

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        return x.mean(dim=(2, 3))  # (B, C)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# InceptionNeXt for image classification
# ---------------------------------------------------------------------------


class InceptionNeXtForImageClassification(PretrainedModel):
    r"""InceptionNeXt with an MLP classification head (Yu et al., 2024).

    Wraps the same trunk as :class:`InceptionNeXt` (patchify stem +
    four stages of MetaNeXt blocks) and adds the reference-recipe MLP
    classifier head: global average pool → ``Linear(dim, 3*dim)`` →
    GELU → LayerNorm → ``Linear(3*dim, num_classes)``:

    .. math::

        \text{logits} = W_2\,
            \mathrm{LN}\!\bigl(\mathrm{GELU}(W_1\,
            \mathrm{GAP}(z^{L}))\bigr) + b_2.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : InceptionNeXtConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See
        :class:`InceptionNeXtConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Patchify stem: :math:`4 \times 4` stride-4 Conv2d + BN.
    stages : nn.ModuleList
        Four MetaNeXt stages.
    head : _MlpClassifierHead
        MLP head: ``fc1 → GELU → LayerNorm → fc2``.

    Notes
    -----
    Reference: Weihao Yu *et al.*, *"InceptionNeXt: When Inception
    Meets ConvNeXt"*, CVPR 2024.  InceptionNeXt-T matches or exceeds
    ConvNeXt-T accuracy at noticeably lower wall-clock latency.

    Examples
    --------
    End-to-end inference with the default InceptionNeXt-T classifier:

    >>> import lucid
    >>> from lucid.models.vision.inception_next import (
    ...     InceptionNeXtConfig, InceptionNeXtForImageClassification,
    ... )
    >>> model = InceptionNeXtForImageClassification(InceptionNeXtConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[InceptionNeXtConfig]] = InceptionNeXtConfig
    base_model_prefix: ClassVar[str] = "inception_next"

    def __init__(self, config: InceptionNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, _, out_dim = _build_inception_next(config)
        self.stem = stem
        self.stages = stages
        self.head = _MlpClassifierHead(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        logits = cast(Tensor, self.head(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
