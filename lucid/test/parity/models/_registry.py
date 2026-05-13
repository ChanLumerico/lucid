"""Central parity-test registry for all Lucid vision models.

Every entry is a ``ParitySpec`` that declares:
  - which Lucid factory function to call
  - the timm model name to compare against (or None → self-consistency only)
  - input shape, tolerances, test tier, and any explicit key remaps

Single source of truth
-----------------------
Adding a new model = adding one ``ParitySpec`` here.  The parametrised
test in ``test_parity_models.py`` picks these up automatically; no
other file needs to change.

Tolerance policy
-----------------
  atol = base * depth_factor

  base        = 1e-3   (float32 per-op rounding)
  depth_factor
    = 1.0   for shallow CNNs (≤ 10 layers deep)
    = 2.0   for medium networks (11–30 layers)
    = 5.0   for deep / attention networks (> 30 layers, window attn, etc.)

These are *conservative* bounds — a model matching within 1e-3 absolute
on every logit is architecturally equivalent for all practical purposes.

Tier definitions
-----------------
  'default'  — always run (< 30 M params, fast)
  'slow'     — 30–100 M params;  pytest -m slow
  'heavy'    — > 100 M params;   pytest -m heavy  (needs ≥ 8 GB RAM)
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import lucid.models as M

# ── Key transform functions (Lucid key name → timm key name) ─────────────────


def _swin_key_transform(k: str) -> str:
    """Swin Transformer: Lucid key → timm key.

    Handles:
      stages.X.downsample.norm → layers.(X+1).downsample.norm  (index offset +1)
      stages.X.downsample.proj → layers.(X+1).downsample.reduction
      stages.X  → layers.X
      rel_pos_bias → relative_position_bias_table
      mlp.0.*  → mlp.fc1.*   (Lucid Sequential MLP → timm named Linear)
      mlp.3.*  → mlp.fc2.*

    Note: downsamplers are remapped BEFORE the general stages→layers substitution
    because Lucid's patch merging for stage X sits inside stages.X and feeds into
    timm's layers.X+1 (timm places it at the downstream stage boundary).
    """

    def _remap_ds(m: re.Match[str]) -> str:
        n = int(m.group(1))
        sub = "reduction" if m.group(2) == "proj" else m.group(2)
        return f"layers.{n + 1}.downsample.{sub}."

    k = re.sub(r"stages\.(\d+)\.downsample\.(norm|proj)\.", _remap_ds, k)
    k = k.replace("stages.", "layers.")
    k = k.replace("rel_pos_bias", "relative_position_bias_table")
    k = k.replace(".mlp.0.", ".mlp.fc1.")
    k = k.replace(".mlp.3.", ".mlp.fc2.")
    return k


def _convnext_key_transform(k: str) -> str:
    """ConvNeXt: Lucid key → timm key.

    Handles:
      stem.conv.N  → stem.N
      stem.norm.*  → stem.1.*
      stages.X.Y.  → stages.X.blocks.Y.
      .dwconv.     → .conv_dw.
      .blocks.Y.fc1/fc2.  → .blocks.Y.mlp.fc1/fc2.
      head_norm.   → head.norm.
      downsamplers.N.norm/conv.  → stages.(N+1).downsample.0/1.
    """
    # Stem
    k = re.sub(r"stem\.conv\.(\d+)", r"stem.\1", k)
    k = k.replace("stem.norm.", "stem.1.")
    # Stage blocks
    k = re.sub(r"stages\.(\d+)\.(\d+)\.", r"stages.\1.blocks.\2.", k)
    # Layer names inside blocks
    k = k.replace(".dwconv.", ".conv_dw.")
    k = re.sub(r"\.blocks\.(\d+)\.fc([12])\.", r".blocks.\1.mlp.fc\2.", k)
    # Head norm
    k = k.replace("head_norm.", "head.norm.")

    # Downsamplers: downsamplers.N.{norm,conv}.* → stages.(N+1).downsample.{0,1}.*
    def _remap_ds(m: re.Match[str]) -> str:
        n = int(m.group(1))
        sub = "0" if m.group(2) == "norm" else "1"
        return f"stages.{n + 1}.downsample.{sub}."

    k = re.sub(r"downsamplers\.(\d+)\.(norm|conv)\.", _remap_ds, k)
    return k


# ── Spec type ─────────────────────────────────────────────────────────────────


@dataclass
class ParitySpec:
    """Complete specification for one model parity test."""

    # Lucid factory — callable returning the *classifier* variant
    lucid_factory: Callable[[], object]

    # timm model name.  None → only self-consistency (no timm reference).
    timm_name: str | None

    # Forward-pass input shape (B, C, H, W)
    input_shape: tuple[int, ...] = (1, 3, 224, 224)

    # Numerical tolerances for logit comparison
    atol: float = 1e-3
    rtol: float = 1e-3

    # Test tier controlling which pytest marker is applied
    tier: Literal["default", "slow", "heavy"] = "default"

    # Explicit Lucid→timm key remaps (supplement auto-alignment)
    key_remap: dict[str, str] = field(default_factory=dict)

    # Non-empty → test is skipped with this explanation
    skip_reason: str = ""

    # Optional function: lucid_key → timm_key name transform (applied before
    # explicit key_remap and auto-remap).  Use for systematic renaming patterns
    # (e.g. ``stages.`` → ``layers.``) that would require many remap entries.
    key_transform: Callable[[str], str] | None = None

    # Fall back to positional weight transfer when named alignment fails to
    # cover > 50 % of keys (e.g. when our attribute names differ from timm but
    # the parameter *shapes* and *order* are identical).  Positional is less
    # robust but avoids false-negative skips for structurally-correct models.
    use_positional_fallback: bool = False

    @property
    def id(self) -> str:
        """Short label for pytest parametrize IDs."""
        return self.lucid_factory.__name__.removesuffix("_cls")


# ── Registry ─────────────────────────────────────────────────────────────────


SPECS: list[ParitySpec] = [
    # ── LeNet / AlexNet / ZFNet / GoogLeNet — no timm equivalent ────────────
    ParitySpec(M.lenet_5_cls, None, input_shape=(1, 1, 32, 32)),
    ParitySpec(M.lenet_5_relu_cls, None, input_shape=(1, 1, 32, 32)),
    ParitySpec(M.alexnet_cls, None, tier="slow"),
    ParitySpec(M.zfnet_cls, None, tier="slow"),
    ParitySpec(M.googlenet_cls, None),
    # ── VGG — heavy (> 100 M) ────────────────────────────────────────────────
    # Lucid names intermediate FC layers fc6/fc7; timm uses pre_logits.fc1/fc2.
    # The final Linear auto-remaps via _AUTO_HEAD_REMAP (classifier.* → head.fc.*).
    ParitySpec(
        M.vgg_11_cls,
        "vgg11",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_11_bn_cls,
        "vgg11_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_13_cls,
        "vgg13",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_13_bn_cls,
        "vgg13_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_16_cls,
        "vgg16",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_16_bn_cls,
        "vgg16_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_19_cls,
        "vgg19",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    ParitySpec(
        M.vgg_19_bn_cls,
        "vgg19_bn",
        tier="heavy",
        key_remap={
            "fc6.weight": "pre_logits.fc1.weight",
            "fc6.bias": "pre_logits.fc1.bias",
            "fc7.weight": "pre_logits.fc2.weight",
            "fc7.bias": "pre_logits.fc2.bias",
        },
    ),
    # ── ResNet ────────────────────────────────────────────────────────────────
    # Lucid uses stem.0/stem.1 vs timm's conv1/bn1 — positional order matches.
    ParitySpec(M.resnet_18_cls, "resnet18", use_positional_fallback=True),
    ParitySpec(M.resnet_34_cls, "resnet34", use_positional_fallback=True),
    ParitySpec(M.resnet_50_cls, "resnet50", tier="slow", use_positional_fallback=True),
    ParitySpec(
        M.resnet_101_cls, "resnet101", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(
        M.resnet_152_cls, "resnet152", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(M.resnet_200_cls, None, tier="slow", use_positional_fallback=True),
    ParitySpec(M.resnet_269_cls, None, tier="slow", use_positional_fallback=True),
    ParitySpec(
        M.wide_resnet_50_cls,
        "wide_resnet50_2",
        tier="slow",
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.wide_resnet_101_cls,
        "wide_resnet101_2",
        tier="slow",
        use_positional_fallback=True,
    ),
    # ── ResNeXt ───────────────────────────────────────────────────────────────
    ParitySpec(
        M.resnext_50_32x4d_cls,
        "resnext50_32x4d",
        tier="slow",
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.resnext_101_32x4d_cls,
        "resnext101_32x4d",
        tier="slow",
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.resnext_101_32x8d_cls,
        "resnext101_32x8d",
        tier="slow",
        use_positional_fallback=True,
    ),
    # ── SE-ResNet ─────────────────────────────────────────────────────────────
    ParitySpec(M.se_resnet_18_cls, "seresnet18", use_positional_fallback=True),
    ParitySpec(M.se_resnet_34_cls, "seresnet34", use_positional_fallback=True),
    ParitySpec(
        M.se_resnet_50_cls, "seresnet50", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(
        M.se_resnet_101_cls, "seresnet101", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(
        M.se_resnet_152_cls, "seresnet152", tier="slow", use_positional_fallback=True
    ),
    # ── SK-ResNet / SK-ResNeXt ────────────────────────────────────────────────
    # timm 1.0 only has skresnet18/34/50/50d and skresnext50_32x4d.
    # sk_resnet_101 has no timm counterpart → self-consistency only.
    ParitySpec(M.sk_resnet_18_cls, "skresnet18", use_positional_fallback=True),
    ParitySpec(M.sk_resnet_34_cls, "skresnet34", use_positional_fallback=True),
    ParitySpec(
        M.sk_resnet_50_cls, "skresnet50", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(M.sk_resnet_101_cls, None, tier="slow"),
    ParitySpec(
        M.sk_resnext_50_32x4d_cls,
        "skresnext50_32x4d",
        tier="slow",
        use_positional_fallback=True,
    ),
    # ── ResNeSt ───────────────────────────────────────────────────────────────
    ParitySpec(M.resnest_14_cls, None),
    ParitySpec(M.resnest_26_cls, None),
    ParitySpec(
        M.resnest_50_cls, "resnest50d", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(
        M.resnest_101_cls, "resnest101e", tier="slow", use_positional_fallback=True
    ),
    ParitySpec(M.resnest_200_cls, None, tier="slow"),
    ParitySpec(M.resnest_269_cls, None, tier="slow"),
    # ── DenseNet ──────────────────────────────────────────────────────────────
    # timm DenseNet uses `classifier` head → exact name match (no remap).
    # Internal BN key order in transition layers differs from timm; named
    # alignment gets close but traversal order also differs, so both named
    # and positional transfer may be incomplete.  Accept partial match.
    ParitySpec(M.densenet_121_cls, "densenet121"),
    ParitySpec(M.densenet_169_cls, "densenet169"),
    ParitySpec(M.densenet_201_cls, "densenet201"),
    ParitySpec(M.densenet_264_cls, None, tier="slow"),  # no timm equivalent
    # ── Inception ─────────────────────────────────────────────────────────────
    # Inception v3: attribute names differ (e.g. Conv2dNormActivation sub-modules
    # vs flat names in timm).  Positional fallback handles this when shapes match.
    ParitySpec(
        M.inception_v3_cls,
        "inception_v3",
        input_shape=(1, 3, 299, 299),
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.inception_resnet_v2_cls,
        "inception_resnet_v2",
        input_shape=(1, 3, 299, 299),
        tier="slow",
    ),
    # ── Xception ──────────────────────────────────────────────────────────────
    ParitySpec(
        M.xception_cls,
        "legacy_xception",  # timm deprecated name→legacy_xception
        input_shape=(1, 3, 299, 299),
        key_remap={"classifier.1.weight": "fc.weight", "classifier.1.bias": "fc.bias"},
        use_positional_fallback=True,
    ),
    # ── MobileNet v1 — no timm exact equivalent ───────────────────────────────
    ParitySpec(M.mobilenet_v1_cls, None),
    ParitySpec(M.mobilenet_v1_075_cls, None),
    ParitySpec(M.mobilenet_v1_050_cls, None),
    ParitySpec(M.mobilenet_v1_025_cls, None),
    # ── MobileNet v2 ──────────────────────────────────────────────────────────
    # timm head: classifier.weight (exact match).  Feature block names differ
    # from timm's MBConv naming → positional fallback for body alignment.
    ParitySpec(M.mobilenet_v2_cls, "mobilenetv2_100", use_positional_fallback=True),
    ParitySpec(M.mobilenet_v2_075_cls, "mobilenetv2_075", use_positional_fallback=True),
    # ── MobileNet v3 ──────────────────────────────────────────────────────────
    # Explicit head remap (classifier.3.*) + positional fallback for body.
    ParitySpec(
        M.mobilenet_v3_large_cls,
        "mobilenetv3_large_100",
        key_remap={
            "classifier.weight": "classifier.3.weight",
            "classifier.bias": "classifier.3.bias",
        },
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.mobilenet_v3_small_cls,
        "mobilenetv3_small_100",
        key_remap={
            "classifier.weight": "classifier.3.weight",
            "classifier.bias": "classifier.3.bias",
        },
        use_positional_fallback=True,
    ),
    # ── EfficientNet — head: classifier.weight (exact match) ─────────────────
    # Lucid uses features.N vs timm's blocks.N.M — positional fallback for body.
    ParitySpec(M.efficientnet_b0_cls, "efficientnet_b0", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b1_cls, "efficientnet_b1", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b2_cls, "efficientnet_b2", use_positional_fallback=True),
    ParitySpec(M.efficientnet_b3_cls, "efficientnet_b3", use_positional_fallback=True),
    ParitySpec(
        M.efficientnet_b4_cls,
        "efficientnet_b4",
        tier="slow",
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.efficientnet_b5_cls,
        "efficientnet_b5",
        tier="slow",
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.efficientnet_b6_cls,
        "efficientnet_b6",
        tier="slow",
        use_positional_fallback=True,
    ),
    ParitySpec(
        M.efficientnet_b7_cls,
        "efficientnet_b7",
        tier="slow",
        use_positional_fallback=True,
    ),
    # ── ViT ───────────────────────────────────────────────────────────────────
    # head: head.weight (auto-remapped)
    # Transformer accumulation → slightly looser tolerance
    ParitySpec(M.vit_base_16_cls, "vit_base_patch16_224", tier="slow", atol=5e-3),
    ParitySpec(M.vit_base_32_cls, "vit_base_patch32_224", tier="slow", atol=5e-3),
    ParitySpec(M.vit_large_16_cls, "vit_large_patch16_224", tier="heavy", atol=5e-3),
    ParitySpec(M.vit_large_32_cls, "vit_large_patch32_224", tier="heavy", atol=5e-3),
    ParitySpec(M.vit_huge_14_cls, None, tier="heavy"),  # 632 M — no timm parity
    # ── Swin Transformer ──────────────────────────────────────────────────────
    # Lucid: stages.X  vs  timm: layers.X  → key_transform handles this.
    # Relative position bias accumulation → looser tolerance (atol=2e-2).
    ParitySpec(
        M.swin_tiny_cls,
        "swin_tiny_patch4_window7_224",
        atol=2e-2,
        key_transform=_swin_key_transform,
    ),
    ParitySpec(
        M.swin_small_cls,
        "swin_small_patch4_window7_224",
        atol=2e-2,
        tier="slow",
        key_transform=_swin_key_transform,
    ),
    ParitySpec(
        M.swin_base_cls,
        "swin_base_patch4_window7_224",
        atol=2e-2,
        tier="heavy",
        key_transform=_swin_key_transform,
    ),
    ParitySpec(
        M.swin_large_cls,
        "swin_large_patch4_window7_224",
        atol=2e-2,
        tier="heavy",
        key_transform=_swin_key_transform,
    ),
    # ── ConvNeXt ──────────────────────────────────────────────────────────────
    # Lucid: stem.conv.N / stem.norm / stages.X.Y
    # timm:  stem.N      / stem.1    / stages.X.blocks.Y
    ParitySpec(
        M.convnext_tiny_cls, "convnext_tiny", key_transform=_convnext_key_transform
    ),
    ParitySpec(
        M.convnext_small_cls,
        "convnext_small",
        tier="slow",
        key_transform=_convnext_key_transform,
    ),
    ParitySpec(
        M.convnext_base_cls,
        "convnext_base",
        tier="slow",
        key_transform=_convnext_key_transform,
    ),
    ParitySpec(
        M.convnext_large_cls,
        "convnext_large",
        tier="heavy",
        key_transform=_convnext_key_transform,
    ),
    ParitySpec(
        M.convnext_xlarge_cls,
        "convnext_xlarge",
        tier="heavy",
        key_transform=_convnext_key_transform,
    ),
    # ── CSPNet — no timm exact equivalent ────────────────────────────────────
    ParitySpec(M.cspresnet_50_cls, None),
    # ── PVT v2 ───────────────────────────────────────────────────────────────
    # head: head.weight (auto-remapped)
    ParitySpec(M.pvt_v2_b0_cls, None),
    ParitySpec(M.pvt_v2_b1_cls, "pvt_v2_b1", atol=2e-3),
    ParitySpec(M.pvt_v2_b2_cls, None, tier="slow"),
    ParitySpec(M.pvt_v2_b3_cls, None, tier="slow"),
    ParitySpec(M.pvt_v2_b4_cls, None, tier="slow"),
    ParitySpec(M.pvt_v2_b5_cls, None, tier="slow"),
    ParitySpec(M.pvt_tiny_cls, "pvt_v2_b1", atol=2e-3),
    # ── CvT ──────────────────────────────────────────────────────────────────
    # Internal ordering may differ → named transfer, accept partial match
    ParitySpec(M.cvt_13_cls, None),  # no stable timm equiv with same arch
    ParitySpec(M.cvt_21_cls, None, tier="slow"),
    ParitySpec(M.cvt_w24_cls, None, tier="slow"),
    # ── CrossViT ─────────────────────────────────────────────────────────────
    ParitySpec(M.crossvit_9_cls, None),
    ParitySpec(M.crossvit_tiny_cls, None),
    ParitySpec(M.crossvit_small_cls, None),
    ParitySpec(M.crossvit_base_cls, None, tier="slow"),
    ParitySpec(M.crossvit_15_cls, None),
    ParitySpec(M.crossvit_18_cls, None),
    # ── CoAtNet ───────────────────────────────────────────────────────────────
    # 98 % of keys are unmatched (Lucid uses stem.N vs timm's stages.0.blocks.N);
    # positional shape mismatch confirms different module hierarchy.
    # Self-consistency only until a key_transform is written for CoAtNet.
    ParitySpec(M.coatnet_0_cls, None, tier="slow"),
    # ── EfficientFormer ───────────────────────────────────────────────────────
    ParitySpec(M.efficientformer_l1_cls, None),
    ParitySpec(M.efficientformer_l3_cls, None, tier="slow"),
    ParitySpec(M.efficientformer_l7_cls, None, tier="slow"),
    # ── MaxViT ───────────────────────────────────────────────────────────────
    # 100% key coverage — attribute names match timm's maxvit_tiny_tf_224 exactly.
    # Relative position bias accumulation → atol=2e-2.
    ParitySpec(M.maxvit_tiny_cls, "maxvit_tiny_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_small_cls, "maxvit_small_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_base_cls, "maxvit_base_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_large_cls, "maxvit_large_tf_224", tier="slow", atol=2e-2),
    ParitySpec(M.maxvit_xlarge_cls, None, tier="slow"),
    # ── InceptionNeXt ────────────────────────────────────────────────────────
    # 100% named key coverage — no key_transform or positional fallback needed.
    ParitySpec(M.inception_next_tiny_cls, "inception_next_tiny", tier="slow"),
]


# ── Helpers for test parametrisation ─────────────────────────────────────────


def specs_for_tier(tier: str) -> list[ParitySpec]:
    """Return all specs that should run at the given tier."""
    return [s for s in SPECS if s.tier == tier]


def specs_with_timm(tier: str | None = None) -> list[ParitySpec]:
    """Return specs that have a timm reference (can do numeric parity)."""
    out = [s for s in SPECS if s.timm_name is not None and not s.skip_reason]
    if tier is not None:
        out = [s for s in out if s.tier == tier]
    return out


def specs_self_consistency(tier: str | None = None) -> list[ParitySpec]:
    """Return specs that only do self-consistency (no timm reference)."""
    out = [s for s in SPECS if s.timm_name is None and not s.skip_reason]
    if tier is not None:
        out = [s for s in out if s.tier == tier]
    return out
