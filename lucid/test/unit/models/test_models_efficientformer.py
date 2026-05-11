"""Unit tests for EfficientFormer (Li et al., 2022)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.efficientformer import (
    EfficientFormerConfig,
    EfficientFormer,
    EfficientFormerForImageClassification,
    efficientformer_l1,
    efficientformer_l1_cls,
)


class TestEfficientFormerConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = EfficientFormerConfig()
        self.assertEqual(cfg.model_type, "efficientformer")


class TestEfficientFormerBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = efficientformer_l1()
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


class TestEfficientFormerClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = efficientformer_l1_cls()
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
        m = EfficientFormerForImageClassification(EfficientFormerConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestEfficientFormerRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="efficientformer")
        self.assertIn("efficientformer_l1", names)
        self.assertIn("efficientformer_l1_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("efficientformer_l1")
        self.assertIsInstance(m, EfficientFormer)


if __name__ == "__main__":
    unittest.main()
