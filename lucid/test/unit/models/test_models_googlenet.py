"""Unit tests for GoogLeNet / Inception v1 (Szegedy et al., 2014)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.googlenet import (
    GoogLeNet,
    GoogLeNetConfig,
    GoogLeNetForImageClassification,
    GoogLeNetOutput,
    googlenet,
    googlenet_cls,
)


class TestGoogLeNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = GoogLeNetConfig()
        self.assertEqual(cfg.model_type, "googlenet")
        self.assertEqual(cfg.num_classes, 1000)
        self.assertTrue(cfg.aux_logits)
        self.assertAlmostEqual(cfg.dropout, 0.4)
        self.assertAlmostEqual(cfg.aux_dropout, 0.7)

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = GoogLeNetConfig(num_classes=100, aux_logits=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = GoogLeNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 100)
            self.assertFalse(cfg2.aux_logits)
        finally:
            os.unlink(path)


class TestGoogLeNetParamCounts(unittest.TestCase):

    def test_backbone_params(self) -> None:
        self.assertEqual(googlenet().num_parameters(), 5_973_552)

    def test_classifier_params(self) -> None:
        # Paper-exact: 13,378,280 (backbone + head + 2 aux classifiers)
        self.assertEqual(googlenet_cls().num_parameters(), 13_378_280)

    def test_no_aux_fewer_params(self) -> None:
        m = GoogLeNetForImageClassification(GoogLeNetConfig(aux_logits=False))
        # Without aux: backbone + head only
        self.assertLess(m.num_parameters(), 13_378_280)


class TestGoogLeNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = googlenet()
        self.model.eval()

    def test_feature_info_3_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 3)
        self.assertEqual([f.num_channels for f in fi], [480, 832, 1024])
        self.assertEqual([f.reduction for f in fi], [8, 16, 32])

    def test_forward_features_shape_224(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (1, 1024, 1, 1))

    def test_forward_returns_googlenet_output(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, GoogLeNetOutput)
        self.assertEqual(out.logits.shape, (2, 1024, 1, 1))
        self.assertIsNone(out.aux_logits1)
        self.assertIsNone(out.aux_logits2)


class TestGoogLeNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = googlenet_cls()

    def test_eval_no_aux(self) -> None:
        self.model.eval()
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))
        self.assertIsNone(out.aux_logits1)
        self.assertIsNone(out.aux_logits2)
        self.assertIsNone(out.loss)

    def test_train_aux_fires(self) -> None:
        self.model.train()
        x = lucid.randn(2, 3, 224, 224)
        out = self.model(x)
        self.assertIsNotNone(out.aux_logits1)
        self.assertIsNotNone(out.aux_logits2)
        self.assertEqual(out.aux_logits1.shape, (2, 1000))  # type: ignore[union-attr]
        self.assertEqual(out.aux_logits2.shape, (2, 1000))  # type: ignore[union-attr]

    def test_train_loss_includes_aux(self) -> None:
        self.model.train()
        x = lucid.randn(2, 3, 224, 224)
        labels = lucid.tensor([0, 1])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())  # type: ignore[union-attr]

    def test_eval_loss_no_aux_penalty(self) -> None:
        self.model.eval()
        x = lucid.randn(2, 3, 224, 224)
        labels = lucid.tensor([0, 1])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)

    def test_no_aux_logits_config(self) -> None:
        m = GoogLeNetForImageClassification(GoogLeNetConfig(aux_logits=False))
        m.train()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertIsNone(out.aux_logits1)
        self.assertIsNone(out.aux_logits2)

    def test_custom_num_classes(self) -> None:
        m = GoogLeNetForImageClassification(GoogLeNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        out = m(x)
        self.assertEqual(out.logits.shape, (1, 10))


class TestGoogLeNetRegistry(unittest.TestCase):

    def test_registered(self) -> None:
        names = models.list_models()
        self.assertIn("googlenet", names)
        self.assertIn("googlenet_cls", names)

    def test_family_filter(self) -> None:
        g = models.list_models(family="googlenet")
        self.assertEqual(sorted(g), ["googlenet", "googlenet_cls"])

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("googlenet")
        self.assertIsInstance(cfg, GoogLeNetConfig)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("googlenet_cls")
        self.assertIsInstance(m, GoogLeNetForImageClassification)


class TestGoogLeNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = googlenet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = GoogLeNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = googlenet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = GoogLeNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
