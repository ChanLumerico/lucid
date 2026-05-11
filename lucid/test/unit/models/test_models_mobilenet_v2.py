"""Unit tests for MobileNet v2 (Sandler et al., 2018)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.mobilenet_v2 import (
    MobileNetV2Config,
    MobileNetV2,
    MobileNetV2ForImageClassification,
    mobilenet_v2,
    mobilenet_v2_cls,
    mobilenet_v2_075,
    mobilenet_v2_075_cls,
)


class TestMobileNetV2Config(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = MobileNetV2Config()
        self.assertEqual(cfg.model_type, "mobilenet_v2")

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = MobileNetV2Config(num_classes=10, width_mult=0.75)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = MobileNetV2Config.from_dict(d)
            self.assertEqual(cfg2.num_classes, 10)
            self.assertAlmostEqual(cfg2.width_mult, 0.75)
        finally:
            os.unlink(path)


class TestMobileNetV2ParamCounts(unittest.TestCase):

    def test_full_has_more_params_than_075(self) -> None:
        self.assertGreater(
            mobilenet_v2().num_parameters(),
            mobilenet_v2_075().num_parameters(),
        )


class TestMobileNetV2Backbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v2()
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


class TestMobileNetV2Classifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v2_cls()
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

    def test_075_variant(self) -> None:
        m = mobilenet_v2_075_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 1000))

    def test_custom_num_classes(self) -> None:
        m = MobileNetV2ForImageClassification(MobileNetV2Config(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestMobileNetV2Registry(unittest.TestCase):

    def test_4_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="mobilenet_v2")), 4)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="mobilenet_v2")
        self.assertIn("mobilenet_v2", names)
        self.assertIn("mobilenet_v2_cls", names)
        self.assertIn("mobilenet_v2_075", names)
        self.assertIn("mobilenet_v2_075_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("mobilenet_v2")
        self.assertIsInstance(m, MobileNetV2)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("mobilenet_v2_cls")
        self.assertIsInstance(m, MobileNetV2ForImageClassification)


class TestMobileNetV2Serialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = mobilenet_v2_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = MobileNetV2ForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
