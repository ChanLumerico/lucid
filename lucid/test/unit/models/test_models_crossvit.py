"""Unit tests for CrossViT (Chen et al., 2021)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.crossvit import (
    CrossViTConfig,
    CrossViT,
    CrossViTForImageClassification,
    crossvit_9,
    crossvit_9_cls,
)


class TestCrossViTConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = CrossViTConfig()
        self.assertEqual(cfg.model_type, "crossvit")


class TestCrossViTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = crossvit_9()
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


class TestCrossViTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = crossvit_9_cls()
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
        m = CrossViTForImageClassification(CrossViTConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestCrossViTRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="crossvit")
        self.assertIn("crossvit_9", names)
        self.assertIn("crossvit_9_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("crossvit_9")
        self.assertIsInstance(m, CrossViT)


if __name__ == "__main__":
    unittest.main()
