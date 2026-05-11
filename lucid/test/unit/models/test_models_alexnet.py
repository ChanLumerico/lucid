"""Unit tests for AlexNet (Krizhevsky, Sutskever & Hinton, 2012)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.alexnet import (
    AlexNet,
    AlexNetConfig,
    AlexNetForImageClassification,
    alexnet,
    alexnet_cls,
)


class TestAlexNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = AlexNetConfig()
        self.assertEqual(cfg.model_type, "alexnet")
        self.assertEqual(cfg.num_classes, 1000)
        self.assertEqual(cfg.in_channels, 3)
        self.assertAlmostEqual(cfg.dropout, 0.5)

    def test_json_round_trip(self) -> None:
        import json, os
        cfg = AlexNetConfig(num_classes=100, dropout=0.3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = AlexNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 100)
            self.assertAlmostEqual(cfg2.dropout, 0.3)
        finally:
            os.unlink(path)


class TestAlexNetParamCounts(unittest.TestCase):
    """Paper-exact counts for the merged single-stream architecture."""

    def test_backbone_params(self) -> None:
        # Conv1(34944) + Conv2(614656) + Conv3(885120) + Conv4(1327488) + Conv5(884992)
        self.assertEqual(alexnet().num_parameters(), 3_747_200)

    def test_classifier_params(self) -> None:
        # backbone + FC6(37752832) + FC7(16781312) + Out(4097000) = 62378344
        self.assertEqual(alexnet_cls().num_parameters(), 62_378_344)


class TestAlexNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = alexnet()
        self.model.eval()

    def test_feature_info_5_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 5)
        self.assertEqual([f.num_channels for f in fi], [96, 256, 384, 384, 256])

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 256, 6, 6))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (2, 256, 6, 6))


class TestAlexNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = alexnet_cls()
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
        m = AlexNetForImageClassification(AlexNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 10))


class TestAlexNetRegistry(unittest.TestCase):

    def test_registered(self) -> None:
        names = models.list_models()
        self.assertIn("alexnet", names)
        self.assertIn("alexnet_cls", names)

    def test_family_filter(self) -> None:
        self.assertEqual(models.list_models(family="alexnet"), ["alexnet", "alexnet_cls"])

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("alexnet")
        self.assertIsInstance(cfg, AlexNetConfig)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("alexnet_cls")
        self.assertIsInstance(m, AlexNetForImageClassification)


class TestAlexNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = alexnet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = AlexNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = alexnet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = AlexNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
