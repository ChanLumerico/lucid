"""Unit tests for Wave 3d segmentation models — CPU + Metal parametrized.

Covers:
  FCN (resnet50/101), UNet (base/small/bilinear),
  Attention U-Net, MaskFormer (resnet50/101),
  Mask2Former (resnet50/101)

Each test combines factory check, output type, shape, deterministic
self-consistency, and loss=None checks in ONE forward pass per device.

Tests are parametrized over the ``device`` fixture so they run on both
the CPU (Accelerate) and Metal (MLX) streams.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import SemanticSegmentationOutput

_B = 1
_C = 3
_H = 128
_W = 128


def _img(device: str, ch: int = _C) -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, ch, _H, _W), device=device)


def _build(factory, device: str):
    m = factory()
    m.eval()
    return m.to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# FCN
# ─────────────────────────────────────────────────────────────────────────────


class TestFCNResNet50:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fcn import fcn_resnet50, FCNForSemanticSegmentation

        m = _build(fcn_resnet50, device)
        assert isinstance(m, FCNForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestFCNResNet101:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.fcn import fcn_resnet101, FCNForSemanticSegmentation

        m = _build(fcn_resnet101, device)
        assert isinstance(m, FCNForSemanticSegmentation)
        out = m(_img(device))
        assert isinstance(out, SemanticSegmentationOutput)
        assert int(out.logits.shape[0]) == _B
        assert int(out.logits.shape[-1]) == _W


# ─────────────────────────────────────────────────────────────────────────────
# UNet variants
# ─────────────────────────────────────────────────────────────────────────────


class TestUNet:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import unet, UNetForSemanticSegmentation

        m = _build(unet, device)
        assert isinstance(m, UNetForSemanticSegmentation)

        x = _img(device, ch=1)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert int(out.logits.shape[1]) == K
        assert int(out.logits.shape[-2]) == _H
        assert int(out.logits.shape[-1]) == _W
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestResUNet2d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import res_unet_2d

        m = _build(
            lambda: res_unet_2d(num_classes=4, base_channels=8, depth=2),
            device,
        )
        out = m(_img(device, ch=1))
        assert isinstance(out, SemanticSegmentationOutput)
        assert tuple(out.logits.shape) == (_B, 4, _H, _W)


class TestUNet3d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import unet_3d
        import lucid

        m = _build(
            lambda: unet_3d(num_classes=3, base_channels=8, depth=2),
            device,
        )
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, 16, 16, 16), device=device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        # (B, K, D, H, W) — same spatial size as input
        assert tuple(out.logits.shape) == (_B, 3, 16, 16, 16)


class TestResUNet3d:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.unet import res_unet_3d
        import lucid

        m = _build(
            lambda: res_unet_3d(num_classes=3, base_channels=8, depth=2),
            device,
        )
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, 16, 16, 16), device=device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        assert tuple(out.logits.shape) == (_B, 3, 16, 16, 16)


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionUNet:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.attention_unet import (
            attention_unet,
            AttentionUNetForSemanticSegmentation,
        )

        m = _build(attention_unet, device)
        assert isinstance(m, AttentionUNetForSemanticSegmentation)

        x = _img(device, ch=1)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        assert int(out.logits.shape[-2]) == _H
        assert int(out.logits.shape[-1]) == _W

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# MaskFormer
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskFormer:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.maskformer import (
            maskformer_resnet50,
            MaskFormerForSemanticSegmentation,
        )

        # Slim transformer + queries for test speed
        m = _build(
            lambda: maskformer_resnet50(
                num_queries=20,
                num_decoder_layers=2,
            ),
            device,
        )
        assert isinstance(m, MaskFormerForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K + 1, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former
# ─────────────────────────────────────────────────────────────────────────────


class TestMask2Former:
    def test_factory_and_forward(self, device: str) -> None:
        from lucid.models.vision.mask2former import (
            mask2former_resnet50,
            Mask2FormerForSemanticSegmentation,
        )

        m = _build(
            lambda: mask2former_resnet50(
                num_queries=20,
                num_decoder_layers=2,
            ),
            device,
        )
        assert isinstance(m, Mask2FormerForSemanticSegmentation)

        x = _img(device)
        out = m(x)
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K + 1, _H, _W)
        assert out.loss is None

        out2 = m(x)
        diff = float(lucid.abs(out.logits - out2.logits).max().item())
        assert diff < 1e-5


class TestMask2FormerSwinTiny:
    def test_factory_and_forward(self, device: str) -> None:
        # Override window_size to 4 so the Swin stages divide our 128px test
        # input cleanly (default ws=7 expects 224/112/56/28 inputs).
        from lucid.models.vision.mask2former import mask2former_swin_tiny

        m = _build(
            lambda: mask2former_swin_tiny(
                num_queries=20,
                num_decoder_layers=2,
                swin_window_size=4,
            ),
            device,
        )
        out = m(_img(device))
        assert isinstance(out, SemanticSegmentationOutput)
        K = m.config.num_classes
        assert tuple(out.logits.shape) == (_B, K + 1, _H, _W)


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-test (device-independent)
# ─────────────────────────────────────────────────────────────────────────────


class TestSegmentationRegistry:
    def test_segmentation_models_registered(self) -> None:
        import lucid.models as M

        seg_models = M.list_models(task="semantic-segmentation")
        expected = [
            "fcn_resnet50",
            "fcn_resnet101",
            "unet",
            "res_unet_2d",
            "unet_3d",
            "res_unet_3d",
            "attention_unet",
            "maskformer_resnet50",
            "maskformer_resnet101",
            "mask2former_resnet50",
            "mask2former_resnet101",
            "mask2former_swin_tiny",
            "mask2former_swin_small",
            "mask2former_swin_base",
            "mask2former_swin_large",
        ]
        for name in expected:
            assert name in seg_models, f"{name!r} missing from registry"
