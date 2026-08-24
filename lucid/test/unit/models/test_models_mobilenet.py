"""Unit tests for MobileNet v1 (Howard et al., 2017)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.mobilenet import (
    MobileNet,
    MobileNetConfig,
    MobileNetForImageClassification,
    mobilenet,
    mobilenet_025,
    mobilenet_050,
    mobilenet_075,
    mobilenet_cls,
)


class TestMobileNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = MobileNetConfig()
        self.assertEqual(cfg.model_type, "mobilenet")
        self.assertAlmostEqual(cfg.width_mult, 1.0)
        self.assertEqual(cfg.num_classes, 1000)

    def test_json_round_trip(self) -> None:
        import json
        import os

        cfg = MobileNetConfig(width_mult=0.75, num_classes=100)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = MobileNetConfig.from_dict(d)
            self.assertAlmostEqual(cfg2.width_mult, 0.75)
            self.assertEqual(cfg2.num_classes, 100)
        finally:
            os.unlink(path)


class TestMobileNetParamCounts(unittest.TestCase):

    def test_full_model_classifier(self) -> None:
        # Paper-exact: 4,231,976
        self.assertEqual(mobilenet_cls().num_parameters(), 4_231_976)

    def test_full_model_backbone(self) -> None:
        self.assertEqual(mobilenet().num_parameters(), 3_206_976)

    def test_width_scaling_reduces_params(self) -> None:
        p100 = mobilenet().num_parameters()
        p075 = mobilenet_075().num_parameters()
        p050 = mobilenet_050().num_parameters()
        p025 = mobilenet_025().num_parameters()
        self.assertGreater(p100, p075)
        self.assertGreater(p075, p050)
        self.assertGreater(p050, p025)


class TestMobileNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet()
        self.model.eval()

    def test_feature_info_5_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 5)
        self.assertEqual([f.reduction for f in fi], [2, 4, 8, 16, 32])

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 1024, 7, 7))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_width_025_fewer_channels(self) -> None:
        m = mobilenet_025()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m.forward_features(x)
        self.assertEqual(out.shape[0], 1)
        self.assertLess(out.shape[1], 1024)


class TestMobileNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_cls()
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
        m = MobileNetForImageClassification(MobileNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestMobileNetRegistry(unittest.TestCase):

    def test_8_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="mobilenet")), 8)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("mobilenet")
        self.assertIsInstance(cfg, MobileNetConfig)
        self.assertAlmostEqual(cfg.width_mult, 1.0)

    def test_create_model(self) -> None:
        m = models.create_model("mobilenet")
        self.assertIsInstance(m, MobileNet)


class TestMobileNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = mobilenet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = MobileNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = mobilenet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = MobileNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


class TestMobileNetWeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enum(self) -> type:
        from lucid.models.vision.mobilenet import MobileNetWeights

        return MobileNetWeights

    def test_default_alias(self) -> None:
        cls = self._enum()
        self.assertIs(cls.DEFAULT, cls.RA4_E3600_R224_IN1K)

    def test_entry_fields(self) -> None:
        cls = self._enum()
        e = cls.RA4_E3600_R224_IN1K.entry
        self.assertEqual(e.num_classes, 1000)
        # Either the pre-upload placeholder or the final 64-char digest.
        self.assertTrue(e.sha256 == "__PENDING_UPLOAD__" or len(e.sha256) == 64)
        self.assertIn("lucid-dl/mobilenet-v1", e.url)
        self.assertIn("/RA4_E3600_R224_IN1K/", e.url)
        meta = cls.RA4_E3600_R224_IN1K.meta
        self.assertEqual(meta["source"], "timm/mobilenetv1_100.ra4_e3600_r224_in1k")
        self.assertEqual(meta["license"], "apache-2.0")
        self.assertEqual(meta["num_params"], 4_231_976)
        self.assertAlmostEqual(meta["metrics"]["ImageNet-1k"]["acc@1"], 75.4)

    def test_transforms_bicubic_224(self) -> None:
        cls = self._enum()
        tf = cls.RA4_E3600_R224_IN1K.transforms()
        self.assertEqual(tf.crop_size, 224)
        self.assertEqual(tf.resize_size, 256)
        self.assertEqual(tf.interpolation, "bicubic")
        self.assertEqual(tuple(tf.mean), (0.5, 0.5, 0.5))
        self.assertEqual(tuple(tf.std), (0.5, 0.5, 0.5))

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        self.assertIn("RA4_E3600_R224_IN1K", list_pretrained("mobilenet_cls"))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestMobileNetPretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_default(self) -> None:
        m = models.mobilenet_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_string_tag(self) -> None:
        m = models.mobilenet_cls(pretrained="RA4_E3600_R224_IN1K")
        self.assertIsInstance(m, MobileNetForImageClassification)


if __name__ == "__main__":
    unittest.main()
