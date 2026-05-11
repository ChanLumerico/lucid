"""Model parity tests — ResNet family.

Covers:
  ResNet-18/34/50/101/152           (He et al., 2015)
  ResNeXt-50/101 32×4d, 101 32×8d  (Xie et al., 2017)
  SE-ResNet-18/50/101/152           (Hu et al., 2018)
  SK-ResNet-50/101                  (Li et al., 2019)
  ResNeSt-50/101                    (Zhang et al., 2020)

All of these have timm equivalents and verified param counts,
so we run full parity (weight-copy + forward comparison).

Size markers
------------
  ResNet-18/34, SE-18, ResNeXt-50: < 30 M → always run
  ResNet-50/101, SE-50/101, SK-50, ResNeSt-50: 25–50 M → slow
  ResNet-152, SE-152, ResNeXt-101 32×8d: 50–90 M → slow
"""

import pytest

import lucid.models as M
from lucid.test.parity.models._utils import (
    requires_timm,
    run_parity,
    run_self_consistency,
)


# ── ResNet ────────────────────────────────────────────────────────────────────


@requires_timm
class TestResNetParity:

    def test_resnet18_parity(self) -> None:
        run_parity(M.resnet_18_cls(), "resnet18")

    def test_resnet34_parity(self) -> None:
        run_parity(M.resnet_34_cls(), "resnet34")

    @pytest.mark.slow
    def test_resnet50_parity(self) -> None:
        run_parity(M.resnet_50_cls(), "resnet50")

    @pytest.mark.slow
    def test_resnet101_parity(self) -> None:
        run_parity(M.resnet_101_cls(), "resnet101")

    @pytest.mark.slow
    def test_resnet152_parity(self) -> None:
        run_parity(M.resnet_152_cls(), "resnet152")


# ── ResNeXt ───────────────────────────────────────────────────────────────────


@requires_timm
class TestResNeXtParity:

    @pytest.mark.slow
    def test_resnext50_32x4d_parity(self) -> None:
        run_parity(M.resnext_50_32x4d_cls(), "resnext50_32x4d")

    @pytest.mark.slow
    def test_resnext101_32x4d_parity(self) -> None:
        run_parity(M.resnext_101_32x4d_cls(), "resnext101_32x4d")

    @pytest.mark.slow
    def test_resnext101_32x8d_parity(self) -> None:
        run_parity(M.resnext_101_32x8d_cls(), "resnext101_32x8d")


# ── SE-ResNet ─────────────────────────────────────────────────────────────────


@requires_timm
class TestSENetParity:

    def test_se_resnet18_parity(self) -> None:
        run_parity(M.se_resnet_18_cls(), "seresnet18")

    @pytest.mark.slow
    def test_se_resnet50_parity(self) -> None:
        run_parity(M.se_resnet_50_cls(), "seresnet50")

    @pytest.mark.slow
    def test_se_resnet101_parity(self) -> None:
        run_parity(M.se_resnet_101_cls(), "seresnet101")

    @pytest.mark.slow
    def test_se_resnet152_parity(self) -> None:
        run_parity(M.se_resnet_152_cls(), "seresnet152")


# ── SK-ResNet ─────────────────────────────────────────────────────────────────


@requires_timm
class TestSKNetParity:

    @pytest.mark.slow
    def test_sk_resnet50_parity(self) -> None:
        run_parity(M.sk_resnet_50_cls(), "skresnet50")

    @pytest.mark.slow
    def test_sk_resnet101_parity(self) -> None:
        run_parity(M.sk_resnet_101_cls(), "skresnet101")

    @pytest.mark.slow
    def test_sk_resnext50_parity(self) -> None:
        # timm equivalent: skresnext50_32x4d
        run_parity(M.sk_resnext_50_32x4d_cls(), "skresnext50_32x4d")


# ── ResNeSt ───────────────────────────────────────────────────────────────────


@requires_timm
class TestResNeStParity:

    @pytest.mark.slow
    def test_resnest50_parity(self) -> None:
        # timm name: resnest50d (the 'd' variant with deep stem + avg_down)
        run_parity(M.resnest_50_cls(), "resnest50d")

    @pytest.mark.slow
    def test_resnest101_parity(self) -> None:
        run_parity(M.resnest_101_cls(), "resnest101e")
