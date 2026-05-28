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
    coatnet_1,
    coatnet_1_cls,
    coatnet_2,
    coatnet_2_cls,
    coatnet_3,
    coatnet_3_cls,
    coatnet_4,
    coatnet_4_cls,
    coatnet_5,
    coatnet_5_cls,
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

    def test_all_variants_registered(self) -> None:
        names = models.list_models(family="coatnet")
        for n in (
            "coatnet_0", "coatnet_0_cls",
            "coatnet_1", "coatnet_1_cls",
            "coatnet_2", "coatnet_2_cls",
            "coatnet_3", "coatnet_3_cls",
            "coatnet_4", "coatnet_4_cls",
            "coatnet_5", "coatnet_5_cls",
        ):
            self.assertIn(n, names)

    def test_create_model(self) -> None:
        m = models.create_model("coatnet_0")
        self.assertIsInstance(m, CoAtNet)


class TestCoAtNetVariantsBuild(unittest.TestCase):
    """Smoke build + paper-cited param margin check for CoAtNet-1..5.

    coatnet_1/2/3 also run a single forward pass (memory-safe on a
    16 GB host).  coatnet_4 (275M) and coatnet_5 (688M) are *build-only*
    — instantiation alone is non-trivial, so a forward is skipped
    rather than risking OOM on the developer machine.
    """

    def _check_params_within(self, model: object, paper_M: int, tol_M: float = 5.0) -> None:
        n = float(getattr(model, "num_parameters")()) / 1e6
        self.assertLess(
            abs(n - paper_M),
            tol_M,
            f"param count {n:.1f}M deviates from paper {paper_M}M by more than {tol_M}M",
        )

    def test_coatnet_1(self) -> None:
        m = coatnet_1_cls()
        self._check_params_within(m, paper_M=42)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_coatnet_2(self) -> None:
        m = coatnet_2_cls()
        self._check_params_within(m, paper_M=75)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_coatnet_3(self) -> None:
        m = coatnet_3_cls()
        self._check_params_within(m, paper_M=168)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_coatnet_4_build_only(self) -> None:
        # 275M params; forward skipped on 16 GB hosts to avoid OOM risk.
        m = coatnet_4_cls()
        self._check_params_within(m, paper_M=275, tol_M=10.0)

    def test_coatnet_5_build_only(self) -> None:
        # 688M params; forward skipped on 16 GB hosts.
        m = coatnet_5_cls()
        self._check_params_within(m, paper_M=688, tol_M=20.0)

    def test_coatnet_1_backbone_output_shape(self) -> None:
        m = coatnet_1()
        m.eval()
        feat = m.forward_features(lucid.randn(1, 3, 224, 224))
        # last stage dim = 768 for coatnet_1 (same as coatnet_0)
        self.assertEqual(feat.shape, (1, 768, 7, 7))


if __name__ == "__main__":
    unittest.main()
