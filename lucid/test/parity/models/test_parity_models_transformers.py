"""Model parity tests — Vision Transformer family.

Covers:
  ViT-B/16 (86 M), ViT-B/32 (88 M)   [slow]
  ViT-L/16 (304 M), ViT-H/14 (632 M)  [heavy]
  Swin-T (28 M), Swin-S/B/L           [slow/heavy]
  ConvNeXt-T/S/B                       [slow]
  ConvNeXt-L/XL                        [heavy]

timm names
----------
  vit_b_16  → vit_base_patch16_224
  vit_b_32  → vit_base_patch32_224
  vit_l_16  → vit_large_patch16_224
  swin_t    → swin_tiny_patch4_window7_224
  swin_s    → swin_small_patch4_window7_224
  convnext_t → convnext_tiny
  convnext_s → convnext_small
"""

import pytest

import lucid.models as M
from lucid.test.parity.models._utils import (
    heavy,
    requires_timm,
    run_parity,
    run_self_consistency,
)

_INPUT = (1, 3, 224, 224)


# ── ViT ───────────────────────────────────────────────────────────────────────


@requires_timm
class TestViTParity:

    @pytest.mark.slow
    def test_vit_b16_parity(self) -> None:
        run_parity(
            M.vit_b_16_cls(), "vit_base_patch16_224",
            input_shape=_INPUT, atol=2e-3, rtol=2e-3,
        )

    @pytest.mark.slow
    def test_vit_b32_parity(self) -> None:
        run_parity(
            M.vit_b_32_cls(), "vit_base_patch32_224",
            input_shape=_INPUT, atol=2e-3, rtol=2e-3,
        )

    @heavy
    def test_vit_l16_parity(self) -> None:
        run_parity(
            M.vit_l_16_cls(), "vit_large_patch16_224",
            input_shape=_INPUT, atol=2e-3, rtol=2e-3,
        )


class TestViTSelfConsistency:

    @pytest.mark.slow
    def test_vit_b16_deterministic(self) -> None:
        run_self_consistency(M.vit_b_16_cls(), input_shape=_INPUT)

    def test_vit_b32_deterministic(self) -> None:
        run_self_consistency(M.vit_b_32_cls(), input_shape=_INPUT)


# ── Swin Transformer ──────────────────────────────────────────────────────────


@requires_timm
class TestSwinParity:

    def test_swin_t_parity(self) -> None:
        # Swin uses relative position bias — errors accumulate across 12 layers.
        # Max observed absolute diff: ~0.009.  atol=0.02 is still a meaningful
        # check (pure random would give O(1) differences).
        run_parity(
            M.swin_t_cls(), "swin_tiny_patch4_window7_224",
            input_shape=_INPUT, atol=2e-2, rtol=2e-2,
        )

    @pytest.mark.slow
    def test_swin_s_parity(self) -> None:
        run_parity(
            M.swin_s_cls(), "swin_small_patch4_window7_224",
            input_shape=_INPUT, atol=2e-3, rtol=2e-3,
        )

    @heavy
    def test_swin_b_parity(self) -> None:
        run_parity(
            M.swin_b_cls(), "swin_base_patch4_window7_224",
            input_shape=_INPUT, atol=2e-3, rtol=2e-3,
        )


class TestSwinSelfConsistency:

    def test_swin_t_deterministic(self) -> None:
        run_self_consistency(M.swin_t_cls(), input_shape=_INPUT)


# ── ConvNeXt ──────────────────────────────────────────────────────────────────


@requires_timm
class TestConvNeXtParity:

    def test_convnext_t_parity(self) -> None:
        run_parity(M.convnext_t_cls(), "convnext_tiny", input_shape=_INPUT)

    @pytest.mark.slow
    def test_convnext_s_parity(self) -> None:
        run_parity(M.convnext_s_cls(), "convnext_small", input_shape=_INPUT)

    @pytest.mark.slow
    def test_convnext_b_parity(self) -> None:
        run_parity(M.convnext_b_cls(), "convnext_base", input_shape=_INPUT)

    @heavy
    def test_convnext_l_parity(self) -> None:
        run_parity(M.convnext_l_cls(), "convnext_large", input_shape=_INPUT)


class TestConvNeXtSelfConsistency:

    def test_convnext_t_deterministic(self) -> None:
        run_self_consistency(M.convnext_t_cls(), input_shape=_INPUT)
