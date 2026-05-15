"""Unit tests for Xception (Chollet, 2017)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.xception import (
    XceptionConfig,
    Xception,
    XceptionForImageClassification,
    XceptionOutput,
    xception,
    xception_cls,
)


class TestXceptionConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = XceptionConfig()
        self.assertEqual(cfg.model_type, "xception")

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = XceptionConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = XceptionConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestXceptionBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = xception()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 299, 299)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestXceptionClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = xception_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(2, 3, 299, 299)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))

    def test_no_labels_no_loss(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        self.assertIsNone(self.model(x).loss)

    def test_labels_produce_scalar_loss(self) -> None:
        x = lucid.randn(2, 3, 299, 299)
        labels = lucid.tensor([0, 999])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())

    def test_custom_num_classes(self) -> None:
        m = XceptionForImageClassification(XceptionConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 299, 299)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestXceptionRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="xception")
        self.assertIn("xception", names)
        self.assertIn("xception_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("xception")
        self.assertIsInstance(m, Xception)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("xception")
        self.assertIsInstance(cfg, XceptionConfig)


if __name__ == "__main__":
    unittest.main()
