"""Unit tests for MobileNet v1 (Howard et al., 2017)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.mobilenet import (
    MobileNetV1,
    MobileNetV1Config,
    MobileNetV1ForImageClassification,
    mobilenet_v1,
    mobilenet_v1_025,
    mobilenet_v1_050,
    mobilenet_v1_075,
    mobilenet_v1_cls,
)


class TestMobileNetV1Config(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = MobileNetV1Config()
        self.assertEqual(cfg.model_type, "mobilenet_v1")
        self.assertAlmostEqual(cfg.width_mult, 1.0)
        self.assertEqual(cfg.num_classes, 1000)

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = MobileNetV1Config(width_mult=0.75, num_classes=100)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = MobileNetV1Config.from_dict(d)
            self.assertAlmostEqual(cfg2.width_mult, 0.75)
            self.assertEqual(cfg2.num_classes, 100)
        finally:
            os.unlink(path)


class TestMobileNetV1ParamCounts(unittest.TestCase):

    def test_full_model_classifier(self) -> None:
        # Paper-exact: 4,231,976
        self.assertEqual(mobilenet_v1_cls().num_parameters(), 4_231_976)

    def test_full_model_backbone(self) -> None:
        self.assertEqual(mobilenet_v1().num_parameters(), 3_206_976)

    def test_width_scaling_reduces_params(self) -> None:
        p100 = mobilenet_v1().num_parameters()
        p075 = mobilenet_v1_075().num_parameters()
        p050 = mobilenet_v1_050().num_parameters()
        p025 = mobilenet_v1_025().num_parameters()
        self.assertGreater(p100, p075)
        self.assertGreater(p075, p050)
        self.assertGreater(p050, p025)


class TestMobileNetV1Backbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v1()
        self.model.eval()

    def test_feature_info_5_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 5)
        self.assertEqual([f.reduction for f in fi], [2, 4, 8, 16, 32])

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 1024, 1, 1))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_width_025_fewer_channels(self) -> None:
        m = mobilenet_v1_025()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m.forward_features(x)
        self.assertEqual(out.shape[0], 1)
        self.assertLess(out.shape[1], 1024)


class TestMobileNetV1Classifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v1_cls()
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
        m = MobileNetV1ForImageClassification(MobileNetV1Config(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestMobileNetV1Registry(unittest.TestCase):

    def test_8_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="mobilenet")), 8)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("mobilenet_v1")
        self.assertIsInstance(cfg, MobileNetV1Config)
        self.assertAlmostEqual(cfg.width_mult, 1.0)

    def test_create_model(self) -> None:
        m = models.create_model("mobilenet_v1")
        self.assertIsInstance(m, MobileNetV1)


class TestMobileNetV1Serialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = mobilenet_v1_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = MobileNetV1ForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = mobilenet_v1_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = MobileNetV1ForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
