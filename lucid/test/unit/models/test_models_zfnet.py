"""Unit tests for ZFNet (Zeiler & Fergus, 2013)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.zfnet import (
    ZFNetConfig,
    ZFNet,
    ZFNetForImageClassification,
    zfnet,
    zfnet_cls,
)


class TestZFNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = ZFNetConfig()
        self.assertEqual(cfg.model_type, "zfnet")

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = ZFNetConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ZFNetConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestZFNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = zfnet()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape[0], 1)

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestZFNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = zfnet_cls()
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
        m = ZFNetForImageClassification(ZFNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestZFNetRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="zfnet")
        self.assertIn("zfnet", names)
        self.assertIn("zfnet_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("zfnet")
        self.assertIsInstance(m, ZFNet)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("zfnet")
        self.assertIsInstance(cfg, ZFNetConfig)


class TestZFNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = zfnet_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ZFNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
