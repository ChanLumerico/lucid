"""Unit tests for Vision Transformer (Dosovitskiy et al., 2020)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.vit import (
    ViT,
    ViTConfig,
    ViTForImageClassification,
    vit_b_16,
    vit_b_16_cls,
    vit_b_32,
    vit_b_32_cls,
    vit_l_16,
    vit_l_16_cls,
)


class TestViTConfig(unittest.TestCase):

    def test_defaults_b16(self) -> None:
        cfg = ViTConfig()
        self.assertEqual(cfg.model_type, "vit")
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.dim, 768)
        self.assertEqual(cfg.depth, 12)
        self.assertEqual(cfg.num_heads, 12)
        self.assertAlmostEqual(cfg.mlp_ratio, 4.0)

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = ViTConfig(patch_size=32, dim=1024, depth=24, num_heads=16)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ViTConfig.from_dict(d)
            self.assertEqual(cfg2.patch_size, 32)
            self.assertEqual(cfg2.dim, 1024)
            self.assertEqual(cfg2.depth, 24)
        finally:
            os.unlink(path)


class TestViTParamCounts(unittest.TestCase):

    def test_b16_classifier(self) -> None:
        # Reference-exact: 86,567,656
        self.assertEqual(vit_b_16_cls().num_parameters(), 86_567_656)

    def test_b32_classifier(self) -> None:
        # Fewer patches → fewer pos embedding params
        self.assertEqual(vit_b_32_cls().num_parameters(), 88_224_232)

    def test_l16_classifier(self) -> None:
        self.assertEqual(vit_l_16_cls().num_parameters(), 304_326_632)

    def test_larger_variants_more_params(self) -> None:
        self.assertGreater(
            vit_l_16().num_parameters(),
            vit_b_16().num_parameters(),
        )


class TestViTBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vit_b_16()
        self.model.eval()

    def test_feature_info_single_stage(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 1)
        self.assertEqual(fi[0].num_channels, 768)
        self.assertEqual(fi[0].reduction, 16)

    def test_forward_features_cls_token(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        feat = self.model.forward_features(x)
        # CLS token: (B, dim)
        self.assertEqual(feat.shape, (2, 768))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        # Unsqueezed: (B, 1, dim)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))

    def test_b32_different_sequence_length(self) -> None:
        m = vit_b_32()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        # B/32 has 7×7=49 patches vs B/16's 14×14=196 patches
        feat = m.forward_features(x)
        self.assertEqual(feat.shape, (1, 768))  # CLS dim same

    def test_l16_larger_dim(self) -> None:
        m = vit_l_16()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        feat = m.forward_features(x)
        self.assertEqual(feat.shape, (1, 1024))


class TestViTClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = vit_b_16_cls()
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
        cfg = ViTConfig(num_classes=10)
        m = ViTForImageClassification(cfg)
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))

    def test_custom_image_size(self) -> None:
        cfg = ViTConfig(image_size=384, patch_size=16, num_classes=10)
        m = ViTForImageClassification(cfg)
        m.eval()
        x = lucid.randn(1, 3, 384, 384)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestViTRegistry(unittest.TestCase):

    def test_10_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="vit")), 10)

    def test_all_variants_present(self) -> None:
        names = models.list_models(family="vit")
        for v in ["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"]:
            self.assertIn(v, names)
            self.assertIn(f"{v}_cls", names)

    def test_auto_config_b16(self) -> None:
        cfg = models.AutoConfig.from_pretrained("vit_b_16")
        self.assertIsInstance(cfg, ViTConfig)
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.dim, 768)

    def test_create_model(self) -> None:
        m = models.create_model("vit_b_16")
        self.assertIsInstance(m, ViT)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("vit_b_16_cls")
        self.assertIsInstance(m, ViTForImageClassification)


class TestViTSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = vit_b_16_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ViTForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = vit_b_16_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = ViTForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
