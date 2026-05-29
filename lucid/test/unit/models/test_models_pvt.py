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


_SHIPPED = (
    ("pvt_v2_b0_cls", "pvt-v2-b0", "pvt_v2_b0.in1k", 3_666_760),
    ("pvt_v2_b2_cls", "pvt-v2-b2", "pvt_v2_b2.in1k", 25_362_856),
    ("pvt_v2_b3_cls", "pvt-v2-b3", "pvt_v2_b3.in1k", 45_238_696),
    ("pvt_v2_b4_cls", "pvt-v2-b4", "pvt_v2_b4.in1k", 62_556_072),
    ("pvt_v2_b5_cls", "pvt-v2-b5", "pvt_v2_b5.in1k", 81_956_008),
)


class TestPVTv2WeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def _enums(self) -> tuple[type, ...]:
        from lucid.models.weights import (
            PVTv2B0Weights,
            PVTv2B2Weights,
            PVTv2B3Weights,
            PVTv2B4Weights,
            PVTv2B5Weights,
        )

        return (
            PVTv2B0Weights,
            PVTv2B2Weights,
            PVTv2B3Weights,
            PVTv2B4Weights,
            PVTv2B5Weights,
        )

    def test_default_aliases_in1k(self) -> None:
        for cls in self._enums():
            self.assertIs(cls.DEFAULT, cls.IN1K)

    def test_entry_fields(self) -> None:
        for cls, (_fac, slug, src, nparams) in zip(self._enums(), _SHIPPED):
            e = cls.IN1K.entry
            self.assertEqual(e.num_classes, 1000)
            self.assertEqual(len(e.sha256), 64)
            self.assertIn(f"lucid-dl/{slug}", e.url)
            self.assertIn("/IN1K/", e.url)
            self.assertEqual(cls.IN1K.meta["source"], f"timm/{src}")
            self.assertEqual(cls.IN1K.meta["license"], "apache-2.0")
            self.assertEqual(cls.IN1K.meta["num_params"], nparams)

    def test_transforms_bicubic_224(self) -> None:
        for cls in self._enums():
            tf = cls.IN1K.transforms()
            self.assertEqual(tf.crop_size, 224)
            self.assertEqual(tf.resize_size, 249)
            self.assertEqual(tf.interpolation, "bicubic")

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for fac, *_ in _SHIPPED:
            self.assertIn("IN1K", list_pretrained(fac))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestPVTv2PretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_b0_default(self) -> None:
        m = models.pvt_v2_b0_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_b2_string_tag(self) -> None:
        m = models.pvt_v2_b2_cls(pretrained="IN1K")
        self.assertIsInstance(m, PVTForImageClassification)


if __name__ == "__main__":
    unittest.main()
