"""Unit tests for LeNet-5 (LeCun et al., 1998)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.lenet import (
    LeNet,
    LeNetConfig,
    LeNetForImageClassification,
    lenet_5,
    lenet_5_cls,
)


class TestLeNetConfig(unittest.TestCase):

    def test_default_original_variant(self) -> None:
        cfg = LeNetConfig()
        self.assertEqual(cfg.model_type, "lenet")
        self.assertEqual(cfg.num_classes, 10)
        self.assertEqual(cfg.in_channels, 1)
        self.assertEqual(cfg.activation, "tanh")
        self.assertEqual(cfg.pooling, "avg")

    def test_modern_variant(self) -> None:
        cfg = LeNetConfig(activation="relu", pooling="max")
        self.assertEqual(cfg.activation, "relu")
        self.assertEqual(cfg.pooling, "max")

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = LeNetConfig(num_classes=100, in_channels=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = LeNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 100)
            self.assertEqual(cfg2.in_channels, 3)
            self.assertEqual(cfg2.model_type, "lenet")
        finally:
            os.unlink(path)


class TestLeNetParamCounts(unittest.TestCase):
    """Canonical LeNet-5 parameter counts from the 1998 paper."""

    def test_backbone_params(self) -> None:
        # C1(156) + C3(2416) + C5(48120) = 50692
        self.assertEqual(lenet_5().num_parameters(), 50_692)

    def test_classifier_params(self) -> None:
        # backbone + F6(120*84+84=10164) + Out(84*10+10=850) = 61706
        self.assertEqual(lenet_5_cls().num_parameters(), 61_706)

    def test_relu_variant_same_param_count(self) -> None:
        # Activation / pooling type doesn't change param count — built via
        # config override on the canonical factory.
        m = lenet_5(activation="relu", pooling="max")
        m_cls = lenet_5_cls(activation="relu", pooling="max")
        self.assertEqual(m.num_parameters(), 50_692)
        self.assertEqual(m_cls.num_parameters(), 61_706)

    def test_rgb_input_params(self) -> None:
        # in_channels=3 → C1: 6*(3*25+1) = 6*76 = 456
        m = LeNetForImageClassification(LeNetConfig(in_channels=3))
        # C1(456) + C3(2416) + C5(48120) + F6(10164) + Out(850) = 62006
        self.assertEqual(m.num_parameters(), 62_006)


class TestLeNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = lenet_5()
        self.model.eval()

    def test_feature_info_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 3)
        self.assertEqual([f.num_channels for f in fi], [6, 16, 120])
        self.assertEqual([f.reduction for f in fi], [2, 4, 32])

    def test_forward_features_shape_32x32(self) -> None:
        x = lucid.randn(1, 1, 32, 32)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 120, 1, 1))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(2, 1, 32, 32)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_relu_backbone_forward(self) -> None:
        m = lenet_5(activation="relu", pooling="max")
        m.eval()
        x = lucid.randn(1, 1, 32, 32)
        out = m.forward_features(x)
        self.assertEqual(out.shape, (1, 120, 1, 1))


class TestLeNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = lenet_5_cls()
        self.model.eval()

    def test_logits_shape_10_classes(self) -> None:
        x = lucid.randn(4, 1, 32, 32)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (4, 10))

    def test_no_labels_no_loss(self) -> None:
        x = lucid.randn(1, 1, 32, 32)
        out = self.model(x)
        self.assertIsNone(out.loss)

    def test_labels_produce_scalar_loss(self) -> None:
        x = lucid.randn(3, 1, 32, 32)
        labels = lucid.tensor([0, 5, 9])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())

    def test_custom_num_classes(self) -> None:
        m = LeNetForImageClassification(LeNetConfig(num_classes=100))
        m.eval()
        x = lucid.randn(1, 1, 32, 32)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 100))

    def test_reset_classifier(self) -> None:
        self.model.reset_classifier(5)
        x = lucid.randn(1, 1, 32, 32)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (1, 5))


class TestLeNetRegistry(unittest.TestCase):

    def test_all_variants_registered(self) -> None:
        names = models.list_models()
        for n in ["lenet_5", "lenet_5_cls"]:
            self.assertIn(n, names)

    def test_family_filter(self) -> None:
        lenet_models = models.list_models(family="lenet")
        self.assertEqual(len(lenet_models), 2)

    def test_create_model_backbone(self) -> None:
        m = models.create_model("lenet_5")
        self.assertIsInstance(m, LeNet)

    def test_create_model_classifier(self) -> None:
        m = models.create_model("lenet_5_cls")
        self.assertIsInstance(m, LeNetForImageClassification)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("lenet_5")
        self.assertIsInstance(cfg, LeNetConfig)
        self.assertEqual(cfg.activation, "tanh")

    def test_auto_model(self) -> None:
        m = models.AutoModel.from_pretrained("lenet_5")
        self.assertIsInstance(m, LeNet)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("lenet_5_cls")
        self.assertIsInstance(m, LeNetForImageClassification)


class TestLeNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = lenet_5_cls()
        m.eval()
        x = lucid.randn(1, 1, 32, 32)
        logits_before = m(x).logits

        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = LeNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((logits_before - m2(x).logits).abs().max().item())

        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = lenet_5_cls()
        m.eval()
        x = lucid.randn(1, 1, 32, 32)
        logits_before = m(x).logits

        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = LeNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((logits_before - m2(x).logits).abs().max().item())

        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
