"""Unit tests for CvT — Convolutional Vision Transformer (Wu et al., 2021)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.cvt import (
    CvTConfig,
    CvT,
    CvTForImageClassification,
    cvt_13,
    cvt_13_cls,
)


class TestCvTConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = CvTConfig()
        self.assertEqual(cfg.model_type, "cvt")


class TestCvTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = cvt_13()
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


class TestCvTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = cvt_13_cls()
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
        m = CvTForImageClassification(CvTConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestCvTRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="cvt")
        self.assertIn("cvt_13", names)
        self.assertIn("cvt_13_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("cvt_13")
        self.assertIsInstance(m, CvT)


if __name__ == "__main__":
    unittest.main()
