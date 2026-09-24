"""InceptionV4 unit tests.

Kept apart from ``test_models_inception.py`` (Inception v3 and
Inception-ResNet-v2): each family's tests live in their own module.
"""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.inception_v4 import (
    InceptionV4Config,
    InceptionV4,
    InceptionV4ForImageClassification,
    inception_v4,
    inception_v4_cls,
)


class TestInceptionV4Backbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_v4()
        self.model.eval()

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        feat = self.model.forward_features(x)
        self.assertEqual(feat.shape[0], 1)


class TestInceptionV4Classifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = inception_v4_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(1, 3, 299, 299)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_custom_num_classes(self) -> None:
        m = InceptionV4ForImageClassification(InceptionV4Config(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 299, 299)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestInceptionV4Registry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="inception_v4")
        self.assertIn("inception_v4", names)
        self.assertIn("inception_v4_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("inception_v4")
        self.assertIsInstance(m, InceptionV4)


class TestInceptionV4WeightsEnum(unittest.TestCase):
    """Static contract of the Weights enum — no network."""

    def _enum(self) -> type:
        from lucid.models.weights import InceptionV4Weights

        return InceptionV4Weights

    def test_default_alias(self) -> None:
        cls = self._enum()
        self.assertIs(cls.DEFAULT, cls.TF_IN1K)
        self.assertEqual(list(cls.__members__), ["TF_IN1K", "DEFAULT"])

    def test_entry_fields(self) -> None:
        e = self._enum().TF_IN1K.entry
        self.assertEqual(e.num_classes, 1000)
        self.assertEqual(len(e.sha256), 64)
        self.assertIn("lucid-dl/inception-v4/", e.url)
        self.assertIn("/TF_IN1K/", e.url)
        meta = self._enum().TF_IN1K.meta
        self.assertEqual(meta["source"], "timm/inception_v4.tf_in1k")
        self.assertEqual(meta["license"], "apache-2.0")
        self.assertEqual(meta["num_params"], inception_v4_cls().num_parameters())
        # timm's results-imagenet.csv row for this exact tag at 299.
        acc = meta["metrics"]["ImageNet-1k"]
        self.assertAlmostEqual(acc["acc@1"], 80.144)
        self.assertAlmostEqual(acc["acc@5"], 94.982)

    def test_transforms_tf_slim_299(self) -> None:
        tf = self._enum().TF_IN1K.transforms()
        self.assertEqual(tf.crop_size, 299)
        # floor(299 / 0.875) — the source pipeline floors, it does not round.
        self.assertEqual(tf.resize_size, 341)
        self.assertEqual(tf.interpolation, "bicubic")
        self.assertEqual(tuple(tf.mean), (0.5, 0.5, 0.5))
        self.assertEqual(tuple(tf.std), (0.5, 0.5, 0.5))

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        self.assertIn("TF_IN1K", list_pretrained("inception_v4_cls"))

    def test_backbone_pretrained_refused(self) -> None:
        # The checkpoint carries the classifier head; the headless backbone
        # must refuse rather than hand back random weights.
        with self.assertRaises(NotImplementedError):
            inception_v4(pretrained=True)


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestInceptionV4PretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_default(self) -> None:
        m = inception_v4_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 299, 299))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_string_tag(self) -> None:
        m = models.inception_v4_cls(pretrained="TF_IN1K")
        self.assertIsInstance(m, InceptionV4ForImageClassification)


if __name__ == "__main__":
    unittest.main()
