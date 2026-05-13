"""Unit tests for Wave 3d segmentation models.

Covers:
  FCN (resnet50/101), UNet (base/small/bilinear),
  Attention U-Net, MaskFormer (resnet50), Mask2Former (resnet50)

Each test group checks:
  1. Factory construction succeeds.
  2. Output type is SemanticSegmentationOutput.
  3. Logits tensor has correct spatial shape matching input H×W.
  4. Self-consistency: same input → identical output.
  5. loss is None when no targets supplied.
"""

import unittest

import lucid
from lucid._tensor.tensor import Tensor
from lucid.models._output import SemanticSegmentationOutput

_B = 1
_C = 3
_H = 128  # Must be divisible by 32 for stride-32 backbones
_W = 128


def _img() -> Tensor:
    lucid.manual_seed(0)
    return lucid.randn((_B, _C, _H, _W))


def _seg_target(num_classes: int = 21) -> Tensor:
    """(B, H, W) integer segmentation map."""
    return lucid.zeros((_B, _H, _W))


# ─────────────────────────────────────────────────────────────────────────────
# FCN
# ─────────────────────────────────────────────────────────────────────────────


class TestFCNResNet50(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.fcn import fcn_resnet50

        self.model = fcn_resnet50()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.fcn import FCNForSemanticSegmentation

        self.assertIsInstance(self.model, FCNForSemanticSegmentation)

    def test_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, SemanticSegmentationOutput)

    def test_logits_shape(self) -> None:
        out = self.model(_img())
        K = self.model.config.num_classes  # 21
        # logits: (B, K, H, W) — same spatial size as input
        self.assertEqual(tuple(out.logits.shape), (_B, K, _H, _W))

    def test_no_loss_without_target(self) -> None:
        out = self.model(_img())
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


class TestFCNResNet101(unittest.TestCase):

    def test_factory_and_shape(self) -> None:
        from lucid.models.vision.fcn import fcn_resnet101

        m = fcn_resnet101()
        m.eval()
        out = m(_img())
        self.assertIsInstance(out, SemanticSegmentationOutput)
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.logits.shape[-1]), _W)


# ─────────────────────────────────────────────────────────────────────────────
# UNet
# ─────────────────────────────────────────────────────────────────────────────


class TestUNet(unittest.TestCase):

    def _unet_img(self) -> Tensor:
        """UNet default: 1-channel biomedical images."""
        lucid.manual_seed(0)
        return lucid.randn((_B, 1, _H, _W))

    def setUp(self) -> None:
        from lucid.models.vision.unet import unet

        self.model = unet()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.unet import UNetForSemanticSegmentation

        self.assertIsInstance(self.model, UNetForSemanticSegmentation)

    def test_output_type(self) -> None:
        out = self.model(self._unet_img())
        self.assertIsInstance(out, SemanticSegmentationOutput)

    def test_logits_spatial_match(self) -> None:
        out = self.model(self._unet_img())
        # Output spatial size must equal input spatial size
        self.assertEqual(int(out.logits.shape[-2]), _H)
        self.assertEqual(int(out.logits.shape[-1]), _W)

    def test_logits_channels(self) -> None:
        out = self.model(self._unet_img())
        K = self.model.config.num_classes  # 2
        self.assertEqual(int(out.logits.shape[1]), K)

    def test_no_loss_without_target(self) -> None:
        out = self.model(self._unet_img())
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = self._unet_img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


class TestUNetSmall(unittest.TestCase):

    def test_factory_and_shape(self) -> None:
        from lucid.models.vision.unet import unet_small

        m = unet_small()
        m.eval()
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, _H, _W))
        out = m(x)
        self.assertIsInstance(out, SemanticSegmentationOutput)
        self.assertEqual(int(out.logits.shape[-2]), _H)
        self.assertEqual(int(out.logits.shape[-1]), _W)


class TestUNetBilinear(unittest.TestCase):

    def test_factory_and_shape(self) -> None:
        from lucid.models.vision.unet import unet_bilinear

        m = unet_bilinear()
        m.eval()
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, _H, _W))
        out = m(x)
        self.assertIsInstance(out, SemanticSegmentationOutput)
        self.assertEqual(int(out.logits.shape[-2]), _H)
        self.assertEqual(int(out.logits.shape[-1]), _W)


# ─────────────────────────────────────────────────────────────────────────────
# Attention U-Net
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionUNet(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.attention_unet import attention_unet

        self.model = attention_unet()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.attention_unet import (
            AttentionUNetForSemanticSegmentation,
        )

        self.assertIsInstance(self.model, AttentionUNetForSemanticSegmentation)

    def test_output_type(self) -> None:
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, _H, _W))
        out = self.model(x)
        self.assertIsInstance(out, SemanticSegmentationOutput)

    def test_logits_spatial_match(self) -> None:
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, _H, _W))
        out = self.model(x)
        self.assertEqual(int(out.logits.shape[-2]), _H)
        self.assertEqual(int(out.logits.shape[-1]), _W)

    def test_self_consistency(self) -> None:
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, _H, _W))
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_small_variant(self) -> None:
        from lucid.models.vision.attention_unet import attention_unet_small

        m = attention_unet_small()
        m.eval()
        lucid.manual_seed(0)
        x = lucid.randn((_B, 1, _H, _W))
        out = m(x)
        self.assertIsInstance(out, SemanticSegmentationOutput)


# ─────────────────────────────────────────────────────────────────────────────
# MaskFormer
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskFormer(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.maskformer import maskformer_resnet50

        self.model = maskformer_resnet50()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.maskformer import MaskFormerForSemanticSegmentation

        self.assertIsInstance(self.model, MaskFormerForSemanticSegmentation)

    def test_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, SemanticSegmentationOutput)

    def test_logits_shape(self) -> None:
        out = self.model(_img())
        K = self.model.config.num_classes  # 150
        # logits: (B, K+1, H, W) — full spatial resolution
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.logits.shape[1]), K + 1)
        self.assertEqual(int(out.logits.shape[2]), _H)
        self.assertEqual(int(out.logits.shape[3]), _W)

    def test_no_loss_without_target(self) -> None:
        out = self.model(_img())
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former
# ─────────────────────────────────────────────────────────────────────────────


class TestMask2Former(unittest.TestCase):

    def setUp(self) -> None:
        from lucid.models.vision.mask2former import mask2former_resnet50

        self.model = mask2former_resnet50()
        self.model.eval()

    def test_factory_type(self) -> None:
        from lucid.models.vision.mask2former import Mask2FormerForSemanticSegmentation

        self.assertIsInstance(self.model, Mask2FormerForSemanticSegmentation)

    def test_output_type(self) -> None:
        out = self.model(_img())
        self.assertIsInstance(out, SemanticSegmentationOutput)

    def test_logits_shape(self) -> None:
        out = self.model(_img())
        K = self.model.config.num_classes  # 150
        self.assertEqual(int(out.logits.shape[0]), _B)
        self.assertEqual(int(out.logits.shape[1]), K + 1)
        self.assertEqual(int(out.logits.shape[2]), _H)
        self.assertEqual(int(out.logits.shape[3]), _W)

    def test_no_loss_without_target(self) -> None:
        out = self.model(_img())
        self.assertIsNone(out.loss)

    def test_self_consistency(self) -> None:
        x = _img()
        out1 = self.model(x)
        out2 = self.model(x)
        diff = float(lucid.abs(out1.logits - out2.logits).max().item())
        self.assertAlmostEqual(diff, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# Registry smoke-test
# ─────────────────────────────────────────────────────────────────────────────


class TestSegmentationRegistry(unittest.TestCase):

    def test_segmentation_models_registered(self) -> None:
        import lucid.models as M

        seg_models = M.list_models(task="semantic-segmentation")
        expected = [
            "fcn_resnet50",
            "fcn_resnet101",
            "unet",
            "unet_small",
            "unet_bilinear",
            "attention_unet",
            "attention_unet_small",
            "maskformer_resnet50",
            "maskformer_resnet101",
            "mask2former_resnet50",
            "mask2former_resnet101",
        ]
        for name in expected:
            self.assertIn(name, seg_models, f"{name!r} missing from registry")


if __name__ == "__main__":
    unittest.main()
