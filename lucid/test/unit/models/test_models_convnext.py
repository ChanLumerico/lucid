"""Unit tests for ConvNeXt (Liu et al., 2022)."""

import tempfile
import unittest

import lucid
import lucid.models as models
from lucid.models.vision.convnext import (
    ConvNeXt,
    ConvNeXtConfig,
    ConvNeXtForImageClassification,
    convnext_tiny,
    convnext_tiny_cls,
    convnext_small_cls,
    convnext_base_cls,
)


class TestConvNeXtConfig(unittest.TestCase):

    def test_defaults_tiny(self) -> None:
        cfg = ConvNeXtConfig()
        self.assertEqual(cfg.model_type, "convnext")
        self.assertEqual(cfg.depths, (3, 3, 9, 3))
        self.assertEqual(cfg.dims, (96, 192, 384, 768))
        self.assertAlmostEqual(cfg.layer_scale_init, 1e-6)

    def test_tuple_coercion(self) -> None:
        import json, os

        cfg = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            with open(path) as f:
                d = json.load(f)
            cfg2 = ConvNeXtConfig.from_dict(d)
            self.assertIsInstance(cfg2.depths, tuple)
            self.assertIsInstance(cfg2.dims, tuple)
            self.assertEqual(cfg2.depths, (3, 3, 27, 3))
        finally:
            os.unlink(path)


class TestConvNeXtParamCounts(unittest.TestCase):

    def test_tiny_classifier(self) -> None:
        # Reference-exact: 28,589,128
        self.assertEqual(convnext_tiny_cls().num_parameters(), 28_589_128)

    def test_small_classifier(self) -> None:
        self.assertEqual(convnext_small_cls().num_parameters(), 50_223_688)

    def test_base_classifier(self) -> None:
        self.assertEqual(convnext_base_cls().num_parameters(), 88_591_464)

    def test_larger_has_more_params(self) -> None:
        self.assertGreater(
            convnext_small_cls().num_parameters(), convnext_tiny_cls().num_parameters()
        )
        self.assertGreater(
            convnext_base_cls().num_parameters(), convnext_small_cls().num_parameters()
        )


class TestConvNeXtBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = convnext_tiny()
        self.model.eval()

    def test_feature_info_4_stages(self) -> None:
        self.assertEqual(len(self.model.feature_info), 4)
        channels = [fi.num_channels for fi in self.model.feature_info]
        self.assertEqual(channels, [96, 192, 384, 768])

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(1, 3, 224, 224)
        feat = self.model.forward_features(x)
        # (B, 768) after global avg pool
        self.assertEqual(feat.shape, (1, 768))

    def test_forward_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput

        x = lucid.randn(1, 3, 224, 224)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, 768))


class TestConvNeXtClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = convnext_tiny_cls()
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
        m = ConvNeXtForImageClassification(ConvNeXtConfig(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestConvNeXtRegistry(unittest.TestCase):

    def test_10_variants_registered(self) -> None:
        self.assertEqual(len(models.list_models(family="convnext")), 10)

    def test_all_sizes_present(self) -> None:
        names = models.list_models(family="convnext")
        for size in ["tiny", "small", "base", "large", "xlarge"]:
            self.assertIn(f"convnext_{size}", names)
            self.assertIn(f"convnext_{size}_cls", names)

    def test_auto_config(self) -> None:
        cfg = models.AutoConfig.from_pretrained("convnext_tiny")
        self.assertIsInstance(cfg, ConvNeXtConfig)
        self.assertEqual(cfg.depths, (3, 3, 9, 3))

    def test_create_model(self) -> None:
        m = models.create_model("convnext_tiny")
        self.assertIsInstance(m, ConvNeXt)


class TestConvNeXtSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = convnext_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ConvNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = convnext_tiny_cls()
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        before = m(x).logits
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = ConvNeXtForImageClassification.from_pretrained(tmp)
            m2.eval()
            diff = float((before - m2(x).logits).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
