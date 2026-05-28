"""Unit tests for CrossViT (Chen et al., ICCV 2021).

The pre-3.5 implementation was a single-stage toy that didn't match
paper Table 2.  This test file targets the paper-faithful 3-stage
multi-scale rebuild — dual-input 240/224, two-branch CrossViT with
proper MultiScaleBlock fusion at each stage.
"""

import unittest

import lucid
import lucid.models as models
from lucid.models.vision.crossvit import (
    CrossViTConfig,
    CrossViT,
    CrossViTForImageClassification,
    crossvit_9,
    crossvit_9_cls,
    crossvit_tiny_cls,
)


class TestCrossViTConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = CrossViTConfig()
        self.assertEqual(cfg.model_type, "crossvit")
        # Paper canonical defaults (CrossViT-Ti shape).
        self.assertEqual(cfg.image_size, 240)
        self.assertEqual(cfg.patch_sizes, (12, 16))
        self.assertEqual(cfg.embed_dims, (96, 192))
        self.assertEqual(len(cfg.depths), 3)  # K=3 stages
        self.assertEqual(cfg.depths[0], (1, 4, 0))
        self.assertEqual(cfg.num_heads, (3, 3))
        self.assertAlmostEqual(cfg.layer_norm_eps, 1e-6)

    def test_img_scale_dual(self) -> None:
        cfg = CrossViTConfig()
        # small branch full resolution, large branch 224/240.
        self.assertEqual(cfg.img_scale[0], 1.0)
        self.assertAlmostEqual(cfg.img_scale[1], 224.0 / 240.0)


class TestCrossViTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = crossvit_9()
        self.model.eval()

    def test_forward_features_returns_two_cls(self) -> None:
        x = lucid.randn(1, 3, 240, 240)
        cls_s, cls_l = self.model.forward_features(x)
        # CrossViT-9: embed_dims = (128, 256).
        self.assertEqual(cls_s.shape, (1, 128))
        self.assertEqual(cls_l.shape, (1, 256))

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 240, 240)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        # Concatenated CLS tokens: (1, 128 + 256).
        self.assertEqual(out.last_hidden_state.shape, (1, 384))


class TestCrossViTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = crossvit_9_cls()
        self.model.eval()

    def test_logits_shape_1000(self) -> None:
        x = lucid.randn(2, 3, 240, 240)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (2, 1000))

    def test_no_labels_no_loss(self) -> None:
        self.assertIsNone(self.model(lucid.randn(1, 3, 240, 240)).loss)

    def test_labels_produce_scalar_loss(self) -> None:
        x = lucid.randn(2, 3, 240, 240)
        labels = lucid.tensor([0, 999])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())

    def test_custom_num_classes(self) -> None:
        m = CrossViTForImageClassification(
            CrossViTConfig(
                num_classes=10, embed_dims=(128, 256), num_heads=(4, 4)
            )
        )
        m.eval()
        out = m(lucid.randn(1, 3, 240, 240))
        self.assertEqual(out.logits.shape, (1, 10))


class TestCrossViTRegistry(unittest.TestCase):

    def test_all_variants_registered(self) -> None:
        names = models.list_models(family="crossvit")
        for n in (
            "crossvit_tiny", "crossvit_tiny_cls",
            "crossvit_small", "crossvit_small_cls",
            "crossvit_base", "crossvit_base_cls",
            "crossvit_9", "crossvit_9_cls",
            "crossvit_15", "crossvit_15_cls",
            "crossvit_18", "crossvit_18_cls",
        ):
            self.assertIn(n, names)

    def test_create_model(self) -> None:
        self.assertIsInstance(models.create_model("crossvit_tiny"), CrossViT)


class TestCrossViTPaperParamCounts(unittest.TestCase):
    """Paper Table 2 param counts; <2% margin to absorb minor diffs."""

    def _check(self, factory_name: str, paper_M: float) -> None:
        n = float(getattr(models, factory_name)().num_parameters()) / 1e6
        delta = abs(n - paper_M) / paper_M
        self.assertLess(
            delta, 0.02,
            f"{factory_name}: {n:.2f}M deviates {delta*100:.1f}% from paper {paper_M}M",
        )

    def test_tiny(self) -> None:
        self._check("crossvit_tiny_cls", 7.0)

    def test_9(self) -> None:
        self._check("crossvit_9_cls", 8.6)

    def test_small(self) -> None:
        self._check("crossvit_small_cls", 26.7)

    def test_15(self) -> None:
        self._check("crossvit_15_cls", 27.4)

    def test_18(self) -> None:
        self._check("crossvit_18_cls", 43.3)

    def test_base_build_only(self) -> None:
        # 105M params; build + a single param-count check is cheap.
        m = models.crossvit_base_cls()
        n = float(m.num_parameters()) / 1e6
        self.assertLess(abs(n - 105.0) / 105.0, 0.02)


class TestCrossViTKeyParityWithTimm(unittest.TestCase):
    """State-dict key topology should mirror timm's so the converter is trivial."""

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("timm") is not None,
        "timm not installed",
    )
    def test_tiny_key_count_matches_timm(self) -> None:
        import timm

        luc = crossvit_tiny_cls()
        tim = timm.create_model("crossvit_tiny_240.in1k", pretrained=False)
        self.assertEqual(len(luc.state_dict()), len(tim.state_dict()))


if __name__ == "__main__":
    unittest.main()
