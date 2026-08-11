"""InceptionV4 unit tests.

Split out of ``test_models_inception.py``: the v4 family is not implemented
yet, and ``conftest.py`` skips a test file wholesale when one of its imports is
missing — which was also taking the InceptionV3 and Inception-ResNet-v2 tests
down with it.  Keeping v4 in its own module lets the implemented families run.
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


# ---------------------------------------------------------------------------
# Inception-ResNet v2
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
