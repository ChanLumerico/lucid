"""Unit tests for VGG (Simonyan & Zisserman, 2014)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.vgg import (
    VGG,
    VGGConfig,
    VGGForImageClassification,
    vgg_11,
    vgg_13,
    vgg_16,
    vgg_19,
    vgg_16_cls,
    vgg_16_bn_cls,
)


class TestVGGConfig(unittest.TestCase):

    def test_default_is_vgg16(self) -> None:
        cfg = VGGConfig()
        self.assertEqual(cfg.model_type, "vgg")
        self.assertEqual(cfg.arch, (2, 2, 3, 3, 3))
        self.assertFalse(cfg.batch_norm)

    def test_arch_tuple_coercion(self) -> None:
        import json, os

        cfg = VGGConfig(arch=(1, 1, 2, 2, 2))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            self.assertIsInstance(d["arch"], list)
            cfg2 = VGGConfig.from_dict(d)
            self.assertIsInstance(cfg2.arch, tuple)
            self.assertEqual(cfg2.arch, (1, 1, 2, 2, 2))
        finally:
            os.unlink(path)


class TestVGGParamCounts(unittest.TestCase):

    def test_vgg11_backbone(self) -> None:
        self.assertEqual(vgg_11().num_parameters(), 9_220_480)

    def test_vgg13_backbone(self) -> None:
        self.assertEqual(vgg_13().num_parameters(), 9_404_992)

    def test_vgg16_backbone(self) -> None:
        self.assertEqual(vgg_16().num_parameters(), 14_714_688)

    def test_vgg19_backbone(self) -> None:
        self.assertEqual(vgg_19().num_parameters(), 20_024_384)

    def test_vgg16_classifier(self) -> None:
        # Paper-exact: 138,357,544
        self.assertEqual(vgg_16_cls().num_parameters(), 138_357_544)

    def test_vgg16_bn_classifier_adds_bn_params(self) -> None:
        # BN adds weight+bias per channel; more than plain VGG-16
        self.assertGreater(vgg_16_bn_cls().num_parameters(), 138_357_544)


class TestVGGBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vgg_16()
        self.model.eval()

    def test_feature_info_5_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 5)
        self.assertEqual([f.num_channels for f in fi], [64, 128, 256, 512, 512])
        self.assertEqual([f.reduction for f in fi], [2, 4, 8, 16, 32])

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 512, 7, 7))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_batch_norm_variant_forward(self) -> None:
        m = models.vgg_16_bn()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m.forward_features(x)
        self.assertEqual(out.shape, (1, 512, 7, 7))


class TestVGGClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vgg_16_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))

    def test_no_labels_no_loss(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        self.assertIsNone(self.model(x).loss)

    def test_labels_produce_scalar_loss(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        labels = lucid.tensor([0, 999])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())

    def test_custom_num_classes(self) -> None:
        m = VGGForImageClassification(VGGConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestVGGRegistry(unittest.TestCase):

    def test_all_16_variants_registered(self) -> None:
        vgg_models = models.list_models(family="vgg")
        self.assertEqual(len(vgg_models), 16)

    def test_backbone_variants(self) -> None:
        for name in [
            "vgg_11",
            "vgg_13",
            "vgg_16",
            "vgg_19",
            "vgg_11_bn",
            "vgg_13_bn",
            "vgg_16_bn",
            "vgg_19_bn",
        ]:
            self.assertIn(name, models.list_models())

    def test_auto_config_vgg16(self) -> None:
        cfg = models.AutoConfig.from_pretrained("vgg_16")
        self.assertIsInstance(cfg, VGGConfig)
        self.assertEqual(cfg.arch, (2, 2, 3, 3, 3))

    def test_create_model(self) -> None:
        m = models.create_model("vgg_11")
        self.assertIsInstance(m, VGG)


class TestVGGSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        # Use VGG-11 (smallest) to keep test fast
        m = models.vgg_11_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = VGGForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = models.vgg_11_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = VGGForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
