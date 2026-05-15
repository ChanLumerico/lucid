"""Unit tests for CoAtNet (Dai et al., 2021)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.coatnet import (
    CoAtNetConfig,
    CoAtNet,
    CoAtNetForImageClassification,
    coatnet_0,
    coatnet_0_cls,
)


class TestCoAtNetConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = CoAtNetConfig()
        self.assertEqual(cfg.model_type, "coatnet")


class TestCoAtNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = coatnet_0()
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


class TestCoAtNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = coatnet_0_cls()
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
        m = CoAtNetForImageClassification(CoAtNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestCoAtNetRegistry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="coatnet")
        self.assertIn("coatnet_0", names)
        self.assertIn("coatnet_0_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("coatnet_0")
        self.assertIsInstance(m, CoAtNet)


if __name__ == "__main__":
    unittest.main()
