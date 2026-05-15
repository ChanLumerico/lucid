"""Unit tests for ResNeSt (Zhang et al., 2020)."""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.resnest import (
    ResNeStConfig,
    ResNeSt,
    ResNeStForImageClassification,
    resnest_50,
    resnest_50_cls,
    resnest_101,
    resnest_101_cls,
)


class TestResNeStConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = ResNeStConfig()
        self.assertEqual(cfg.model_type, "resnest")

    def test_json_round_trip(self) -> None:
        import json, os, tempfile

        cfg = ResNeStConfig(num_classes=10)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ResNeStConfig.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
        finally:
            os.unlink(path)


class TestResNeStParamCounts(unittest.TestCase):

    def test_101_more_params_than_50(self) -> None:
        self.assertGreater(
            resnest_101().num_parameters(),
            resnest_50().num_parameters(),
        )


class TestResNeStBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = resnest_50()
        self.model.eval()

    def test_feature_info(self) -> None:
        fi = self.model.feature_info
        self.assertGreater(len(fi), 0)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)


class TestResNeStClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = resnest_50_cls()
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
        m = ResNeStForImageClassification(ResNeStConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestResNeStRegistry(unittest.TestCase):

    def test_4_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="resnest")), 4)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="resnest")
        self.assertIn("resnest_50", names)
        self.assertIn("resnest_50_cls", names)
        self.assertIn("resnest_101", names)
        self.assertIn("resnest_101_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("resnest_50")
        self.assertIsInstance(m, ResNeSt)


if __name__ == "__main__":
    unittest.main()
