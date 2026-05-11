"""Model parity tests — newer architecture family.

Covers:
  CSPNet          (Wang et al., 2019)   — 7.5 M    self-consistency
  MaxViT-T        (Tu et al., 2022)     — 30.9 M   slow + timm
  PVT v2-B1       (Wang et al., 2022)   — 14.0 M   timm
  CvT-13          (Wu et al., 2021)     — 20.0 M   self-consistency
  CrossViT-9      (Chen et al., 2021)   — small    self-consistency
  CoAtNet-0       (Dai et al., 2021)    — 26.1 M   slow + timm
  EfficientFormer (Li et al., 2022)     — small    self-consistency
  InceptionNeXt-T (Yu et al., 2023)    — 28.3 M   slow + timm

timm equivalents used when available.
"""

import pytest

import lucid.models as M
from lucid.test.parity.models._utils import (
    requires_timm,
    run_parity,
    run_self_consistency,
)

_INPUT = (1, 3, 224, 224)


# ── CSPNet (no timm exact equivalent) ────────────────────────────────────────


class TestCSPNetSelfConsistency:

    def test_cspresnet50_deterministic(self) -> None:
        run_self_consistency(M.cspresnet_50_cls(), input_shape=_INPUT)


# ── MaxViT ────────────────────────────────────────────────────────────────────


@requires_timm
class TestMaxViTParity:

    @pytest.mark.slow
    def test_maxvit_t_parity(self) -> None:
        # timm: maxvit_tiny_tf_224
        run_parity(
            M.maxvit_t_cls(),
            "maxvit_tiny_tf_224",
            input_shape=_INPUT,
            atol=2e-3, rtol=2e-3,
        )


class TestMaxViTSelfConsistency:

    def test_maxvit_t_deterministic(self) -> None:
        run_self_consistency(M.maxvit_t_cls(), input_shape=_INPUT)


# ── PVT v2 ───────────────────────────────────────────────────────────────────


@requires_timm
class TestPVTParity:

    def test_pvt_tiny_parity(self) -> None:
        # Our pvt_tiny matches pvt_v2_b1 architecture
        run_parity(
            M.pvt_tiny_cls(),
            "pvt_v2_b1",
            input_shape=_INPUT,
            atol=2e-3, rtol=2e-3,
        )


class TestPVTSelfConsistency:

    def test_pvt_tiny_deterministic(self) -> None:
        run_self_consistency(M.pvt_tiny_cls(), input_shape=_INPUT)


# ── CvT ──────────────────────────────────────────────────────────────────────


class TestCvTSelfConsistency:
    # CvT architecture in timm differs in key naming → self-consistency only

    def test_cvt13_deterministic(self) -> None:
        run_self_consistency(M.cvt_13_cls(), input_shape=_INPUT)


# ── CrossViT ─────────────────────────────────────────────────────────────────


class TestCrossViTSelfConsistency:

    def test_crossvit9_deterministic(self) -> None:
        run_self_consistency(M.crossvit_9_cls(), input_shape=_INPUT)


# ── CoAtNet ───────────────────────────────────────────────────────────────────


@requires_timm
class TestCoAtNetParity:

    @pytest.mark.slow
    def test_coatnet0_parity(self) -> None:
        run_parity(
            M.coatnet_0_cls(),
            "coatnet_0_rw_224",
            input_shape=_INPUT,
            atol=2e-3, rtol=2e-3,
        )


class TestCoAtNetSelfConsistency:

    def test_coatnet0_deterministic(self) -> None:
        run_self_consistency(M.coatnet_0_cls(), input_shape=_INPUT)


# ── EfficientFormer ──────────────────────────────────────────────────────────


class TestEfficientFormerSelfConsistency:

    def test_efficientformer_l1_deterministic(self) -> None:
        run_self_consistency(M.efficientformer_l1_cls(), input_shape=_INPUT)


# ── InceptionNeXt ────────────────────────────────────────────────────────────


@requires_timm
class TestInceptionNeXtParity:

    @pytest.mark.slow
    def test_inception_next_t_parity(self) -> None:
        run_parity(
            M.inception_next_t_cls(),
            "inception_next_tiny",
            input_shape=_INPUT,
        )


class TestInceptionNeXtSelfConsistency:

    def test_inception_next_t_deterministic(self) -> None:
        run_self_consistency(M.inception_next_t_cls(), input_shape=_INPUT)
