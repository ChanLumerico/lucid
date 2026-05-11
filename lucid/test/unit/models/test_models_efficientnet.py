"""Unit tests for EfficientNet B0–B7 (Tan & Le, 2019)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.efficientnet import (
    EfficientNet,
    EfficientNetConfig,
    EfficientNetForImageClassification,
    efficientnet_b0,
    efficientnet_b0_cls,
    efficientnet_b3,
    efficientnet_b7,
)


class TestEfficientNetConfig(unittest.TestCase):

    def test_b0_defaults(self) -> None:
        cfg = EfficientNetConfig()
        self.assertEqual(cfg.model_type, "efficientnet")
        self.assertAlmostEqual(cfg.width_mult, 1.0)
        self.assertAlmostEqual(cfg.depth_mult, 1.0)
        self.assertAlmostEqual(cfg.dropout, 0.2)
        self.assertAlmostEqual(cfg.se_ratio, 0.25)

    def test_json_round_trip(self) -> None:
        import json, os

        cfg = EfficientNetConfig(width_mult=1.4, depth_mult=1.8, dropout=0.4)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = EfficientNetConfig.from_dict(d)
            self.assertAlmostEqual(cfg2.width_mult, 1.4)
            self.assertAlmostEqual(cfg2.depth_mult, 1.8)
        finally:
            os.unlink(path)


class TestEfficientNetParamCounts(unittest.TestCase):

    def test_b0_backbone(self) -> None:
        self.assertEqual(efficientnet_b0().num_parameters(), 4_007_548)

    def test_b0_classifier(self) -> None:
        # Reference-exact: 5,288,548
        self.assertEqual(efficientnet_b0_cls().num_parameters(), 5_288_548)

    def test_compound_scaling_increases_params(self) -> None:
        p_b0 = efficientnet_b0().num_parameters()
        p_b3 = efficientnet_b3().num_parameters()
        p_b7 = efficientnet_b7().num_parameters()
        self.assertGreater(p_b3, p_b0)
        self.assertGreater(p_b7, p_b3)


class TestEfficientNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = efficientnet_b0()
        self.model.eval()

    def test_feature_info_7_stages(self) -> None:
        fi = self.model.feature_info
        self.assertEqual(len(fi), 7)

    def test_forward_features_shape_b0(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        out = self.model.forward_features(x)
        # B0 head_ch = _make_divisible(1280 * 1.0) = 1280
        self.assertEqual(out.shape, (1, 1280, 1, 1))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)

    def test_b3_wider_feature_map(self) -> None:
        m = efficientnet_b3()
        m.eval()
        x = lucid.randn(1, 3, 300, 300)
        out = m.forward_features(x)
        self.assertEqual(out.shape[0], 1)
        self.assertGreater(out.shape[1], 1280)  # wider due to width_mult=1.2


class TestEfficientNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = efficientnet_b0_cls()
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
        m = EfficientNetForImageClassification(EfficientNetConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestEfficientNetRegistry(unittest.TestCase):

    def test_16_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="efficientnet")), 16)

    def test_all_b_variants_present(self) -> None:
        names = models.list_models(family="efficientnet")
        for b in range(8):
            self.assertIn(f"efficientnet_b{b}", names)
            self.assertIn(f"efficientnet_b{b}_cls", names)

    def test_auto_config_b0(self) -> None:
        cfg = models.AutoConfig.from_pretrained("efficientnet_b0")
        self.assertIsInstance(cfg, EfficientNetConfig)
        self.assertAlmostEqual(cfg.width_mult, 1.0)

    def test_auto_config_b7_scaling(self) -> None:
        cfg = models.AutoConfig.from_pretrained("efficientnet_b7")
        self.assertAlmostEqual(cfg.width_mult, 2.0)
        self.assertAlmostEqual(cfg.depth_mult, 3.1)

    def test_create_model(self) -> None:
        m = models.create_model("efficientnet_b0")
        self.assertIsInstance(m, EfficientNet)


class TestEfficientNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = efficientnet_b0_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = EfficientNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = efficientnet_b0_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = EfficientNetForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
