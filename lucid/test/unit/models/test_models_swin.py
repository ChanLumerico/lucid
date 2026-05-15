"""Unit tests for Swin Transformer (Liu et al., 2021)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.swin import (
    SwinConfig,
    SwinTransformer,
    SwinTransformerForImageClassification,
    swin_tiny,
    swin_tiny_cls,
    swin_small_cls,
    swin_base_cls,
)


class TestSwinConfig(unittest.TestCase):

    def test_defaults_swin_tiny(self) -> None:
        cfg = SwinConfig()
        self.assertEqual(cfg.model_type, "swin")
        self.assertEqual(cfg.embed_dim, 96)
        self.assertEqual(cfg.depths, (2, 2, 6, 2))
        self.assertEqual(cfg.num_heads, (3, 6, 12, 24))
        self.assertEqual(cfg.window_size, 7)

    def test_tuple_coercion(self) -> None:
        import json, os

        cfg = SwinConfig(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            self.assertIsInstance(d["depths"], list)
            cfg2 = SwinConfig.from_dict(d)
            self.assertIsInstance(cfg2.depths, tuple)
            self.assertIsInstance(cfg2.num_heads, tuple)
            self.assertEqual(cfg2.depths, (2, 2, 18, 2))
        finally:
            os.unlink(path)


class TestSwinParamCounts(unittest.TestCase):

    def test_swin_tiny_classifier(self) -> None:
        # Reference-exact: 28,288,354
        self.assertEqual(swin_tiny_cls().num_parameters(), 28_288_354)

    def test_swin_small_classifier(self) -> None:
        self.assertEqual(swin_small_cls().num_parameters(), 49_606_258)

    def test_swin_base_classifier(self) -> None:
        self.assertEqual(swin_base_cls().num_parameters(), 87_768_224)

    def test_larger_has_more_params(self) -> None:
        self.assertGreater(
            swin_small_cls().num_parameters(), swin_tiny_cls().num_parameters()
        )
        self.assertGreater(
            swin_base_cls().num_parameters(), swin_small_cls().num_parameters()
        )


class TestSwinBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = swin_tiny()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 4)

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        # Swin-T final dim = 96 * 8 = 768
        self.assertEqual(feat.shape, (1, 768))

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))


class TestSwinClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = swin_tiny_cls()
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
        m = SwinTransformerForImageClassification(SwinConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestSwinRegistry(unittest.TestCase):

    def test_8_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="swin")), 8)

    def test_all_sizes_present(self) -> None:
        names = models.list_models(family="swin")
        for size in ["t", "s", "b", "l"]:
            self.assertIn(f"swin_{size}", names)
            self.assertIn(f"swin_{size}_cls", names)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("swin_tiny")
        self.assertIsInstance(cfg, SwinConfig)
        self.assertEqual(cfg.embed_dim, 96)

    def test_create_model(self) -> None:
        m = models.create_model("swin_tiny")
        self.assertIsInstance(m, SwinTransformer)


class TestSwinSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = swin_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = SwinTransformerForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = swin_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = SwinTransformerForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
