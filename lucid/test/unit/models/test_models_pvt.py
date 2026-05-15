"""Unit tests for PVT — Pyramid Vision Transformer (Wang et al., 2021)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.pvt import (
    PVTConfig,
    PVT,
    PVTForImageClassification,
    pvt_tiny,
    pvt_tiny_cls,
)


class TestPVTConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = PVTConfig()
        self.assertEqual(cfg.model_type, "pvt")


class TestPVTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = pvt_tiny()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestPVTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = pvt_tiny_cls()
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
        m = PVTForImageClassification(PVTConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestPVTRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="pvt")
        self.assertIn("pvt_tiny", names)
        self.assertIn("pvt_tiny_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("pvt_tiny")
        self.assertIsInstance(m, PVT)


if __name__ == "__main__":
    unittest.main()
