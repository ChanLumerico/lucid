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
    coatnet_2_cls,
    coatnet_3_cls,
    coatnet_4_cls,
    coatnet_5_cls,
    coatnet_6,
    coatnet_6_cls,
    coatnet_7,
    coatnet_7_cls,
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
            "coatnet_0",
            "coatnet_0_cls",
            "coatnet_1",
            "coatnet_1_cls",
            "coatnet_2",
            "coatnet_2_cls",
            "coatnet_3",
            "coatnet_3_cls",
            "coatnet_4",
            "coatnet_4_cls",
            "coatnet_5",
            "coatnet_5_cls",
            "coatnet_6",
            "coatnet_6_cls",
            "coatnet_7",
            "coatnet_7_cls",
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

    def _check_params_within(
        self, model: object, paper_M: int, tol_M: float = 5.0
    ) -> None:
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


class TestCoAtNetMixedS3Variants(unittest.TestCase):
    """coatnet_6 / coatnet_7 — mixed-S3 (MBConv + Transformer co-existing).

    Instantiating these on a 16 GB host is memory-prohibitive
    (≈ 1.5B / 2.4B params; the parameter tensors alone are 6 / 10 GB
    before activations).  These tests therefore exercise *only* the
    invariants that don't require building the model:

    * the factories are reachable + importable
    * the configs encode the paper §A.2 / Table 12 mixed-S3 spec
    * a *tiny* mixed-S3 config (proportional to coatnet_6 but scaled
      down by ~50x in every dimension) actually instantiates + runs a
      forward pass, exercising the new ``mixed_s3`` code path in
      ``_build_body`` without needing the full footprint.
    """

    def test_factories_importable(self) -> None:
        # Direct factory refs (not via getattr) so coatnet_6/7 are in
        # the test's import-time graph.
        self.assertTrue(callable(coatnet_6))
        self.assertTrue(callable(coatnet_6_cls))
        self.assertTrue(callable(coatnet_7))
        self.assertTrue(callable(coatnet_7_cls))

    def test_coatnet_6_config_matches_paper_table_12(self) -> None:
        from lucid.models.vision.coatnet._pretrained import _CFG_6  # type: ignore[attr-defined]

        self.assertEqual(_CFG_6.variant, "coatnet_6")
        # Paper §A.2: S0=192, S1=192, S2=384, S3-mb=768, S4=2048; S3-attn=1536.
        self.assertEqual(_CFG_6.stem_width, 192)
        self.assertEqual(_CFG_6.dims, (192, 384, 768, 2048))
        self.assertEqual(_CFG_6.mixed_s3, (8, 42, 1536))

    def test_coatnet_7_config_matches_paper_table_12(self) -> None:
        from lucid.models.vision.coatnet._pretrained import _CFG_7  # type: ignore[attr-defined]

        self.assertEqual(_CFG_7.variant, "coatnet_7")
        # Paper §A.2: S0=192, S1=256, S2=512, S3-mb=1024, S4=3072; S3-attn=2048.
        self.assertEqual(_CFG_7.stem_width, 192)
        self.assertEqual(_CFG_7.dims, (256, 512, 1024, 3072))
        self.assertEqual(_CFG_7.mixed_s3, (8, 42, 2048))

    def test_mixed_s3_builder_via_tiny_config(self) -> None:
        # Smoke-build the mixed-S3 code path with a tiny config that has
        # the coatnet_6 *shape* (mixed_s3 set, dims that grow then expand)
        # but scaled down so it fits in test memory.
        cfg = CoAtNetConfig(
            variant="test_mixed_s3",
            blocks_per_stage=(1, 1, 2, 1),
            dims=(16, 32, 64, 128),
            stem_width=16,
            attn_heads=(4, 4),
            mbconv_expand=4,
            head_hidden_size=128,
            mixed_s3=(1, 1, 64),
        )
        m = CoAtNetForImageClassification(cfg)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
