"""Unit tests for ResNet family (config, backbone, classifier, registry, serialization)."""

import tempfile
import unittest
import warnings

import lucid
import lucid.models as models
from lucid.models.vision.resnet import (
    ResNet,
    ResNetConfig,
    ResNetForImageClassification,
    resnet_18,
    resnet_50,
    resnet_50_cls,
)


class TestResNetConfig(unittest.TestCase):

    def test_default_config(self) -> None:
        cfg = ResNetConfig()
        self.assertEqual(cfg.model_type, "resnet")
        self.assertEqual(cfg.num_classes, 1000)
        self.assertEqual(cfg.block_type, "bottleneck")
        self.assertEqual(cfg.layers, (3, 4, 6, 3))

    def test_basic_block_config(self) -> None:
        cfg = ResNetConfig(block_type="basic", layers=(2, 2, 2, 2))
        self.assertEqual(cfg.block_type, "basic")
        self.assertEqual(cfg.layers, (2, 2, 2, 2))

    def test_json_round_trip_coerces_tuples(self) -> None:
        import json, tempfile, os
        cfg = ResNetConfig(block_type="basic", layers=(2, 2, 2, 2))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            # On disk: JSON arrays (lists). from_dict must coerce to tuples.
            with open(path) as f:
                d = json.load(f)
            self.assertIsInstance(d["layers"], list)
            cfg2 = ResNetConfig.from_dict(d)
            self.assertIsInstance(cfg2.layers, tuple)
            self.assertIsInstance(cfg2.hidden_sizes, tuple)
            self.assertEqual(cfg2.layers, (2, 2, 2, 2))
        finally:
            os.unlink(path)

    def test_from_dict_ignores_unknown_fields(self) -> None:
        d = ResNetConfig().to_dict()
        d["future_field"] = 42
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg = ResNetConfig.from_dict(d)
        self.assertTrue(any("future_field" in str(x.message) for x in w))
        self.assertEqual(cfg.model_type, "resnet")


class TestResNetParamCounts(unittest.TestCase):
    """Verify canonical parameter counts match the reference architecture."""

    def test_resnet18_backbone(self) -> None:
        m = resnet_18()
        self.assertEqual(m.num_parameters(), 11_176_512)

    def test_resnet50_backbone(self) -> None:
        m = resnet_50()
        self.assertEqual(m.num_parameters(), 23_508_032)

    def test_resnet50_classifier(self) -> None:
        m = resnet_50_cls()
        # Backbone + Linear(2048, 1000) + bias = 23_508_032 + 2_048_000 + 1_000
        self.assertEqual(m.num_parameters(), 25_557_032)

    def test_resnet18_classifier(self) -> None:
        m = models.create_model("resnet_18_cls")
        # Backbone + Linear(512, 1000) + bias = 11_176_512 + 512_000 + 1_000
        self.assertEqual(m.num_parameters(), 11_689_512)


class TestResNetBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = ResNetConfig(block_type="basic", layers=(2, 2, 2, 2))
        self.model = ResNet(self.cfg)
        self.model.eval()

    def test_feature_info_len(self) -> None:
        self.assertEqual(len(self.model.feature_info), 4)

    def test_feature_info_reductions(self) -> None:
        reductions = [fi.reduction for fi in self.model.feature_info]
        self.assertEqual(reductions, [4, 8, 16, 32])

    def test_feature_info_channels_basic(self) -> None:
        channels = [fi.num_channels for fi in self.model.feature_info]
        self.assertEqual(channels, [64, 128, 256, 512])

    def test_forward_features_shape(self) -> None:
        x = lucid.randn(2, 3, 224, 224)
        out = self.model.forward_features(x)
        self.assertEqual(out.shape, (2, 512, 7, 7))

    def test_forward_returns_base_model_output(self) -> None:
        from lucid.models._output import BaseModelOutput
        x = lucid.randn(1, 3, 64, 64)
        out = self.model(x)
        self.assertIsInstance(out, BaseModelOutput)
        self.assertEqual(len(out.last_hidden_state.shape), 4)

    def test_backbone_feature_info_bottleneck(self) -> None:
        cfg = ResNetConfig(block_type="bottleneck", layers=(3, 4, 6, 3))
        m = ResNet(cfg)
        channels = [fi.num_channels for fi in m.feature_info]
        self.assertEqual(channels, [256, 512, 1024, 2048])


class TestResNetClassifier(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = ResNetConfig(
            block_type="basic", layers=(2, 2, 2, 2), num_classes=10
        )
        self.model = ResNetForImageClassification(self.cfg)
        self.model.eval()

    def test_forward_logits_shape(self) -> None:
        x = lucid.randn(3, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (3, 10))

    def test_forward_no_labels_loss_is_none(self) -> None:
        x = lucid.randn(1, 3, 64, 64)
        out = self.model(x)
        self.assertIsNone(out.loss)

    def test_forward_with_labels_returns_loss(self) -> None:
        x = lucid.randn(2, 3, 64, 64)
        labels = lucid.tensor([0, 1])
        out = self.model(x, labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.shape, ())

    def test_reset_classifier(self) -> None:
        self.model.reset_classifier(5)
        x = lucid.randn(1, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.logits.shape, (1, 5))


class TestResNetRegistry(unittest.TestCase):

    def test_all_variants_registered(self) -> None:
        names = models.list_models()
        for v in ["resnet_18", "resnet_34", "resnet_50", "resnet_101", "resnet_152"]:
            self.assertIn(v, names)
        for v in ["resnet_18_cls", "resnet_34_cls", "resnet_50_cls"]:
            self.assertIn(v, names)

    def test_is_model(self) -> None:
        self.assertTrue(models.is_model("resnet_50"))
        self.assertTrue(models.is_model("resnet-50"))  # hyphen normalization
        self.assertFalse(models.is_model("resnet_999"))

    def test_create_model_backbone(self) -> None:
        m = models.create_model("resnet_18")
        self.assertIsInstance(m, ResNet)

    def test_create_model_cls(self) -> None:
        m = models.create_model("resnet_50_cls")
        self.assertIsInstance(m, ResNetForImageClassification)

    def test_list_models_family_filter(self) -> None:
        resnet_models = models.list_models(family="resnet")
        self.assertTrue(all("resnet" in n for n in resnet_models))
        self.assertGreaterEqual(len(resnet_models), 10)

    def test_auto_config_fast_path(self) -> None:
        cfg = models.AutoConfig.from_pretrained("resnet_50")
        self.assertIsInstance(cfg, ResNetConfig)
        self.assertEqual(cfg.layers, (3, 4, 6, 3))

    def test_auto_model_from_pretrained_base(self) -> None:
        m = models.AutoModel.from_pretrained("resnet_18")
        self.assertIsInstance(m, ResNet)

    def test_auto_model_for_classification(self) -> None:
        m = models.AutoModelForImageClassification.from_pretrained("resnet_50_cls")
        self.assertIsInstance(m, ResNetForImageClassification)


class TestResNetSerialization(unittest.TestCase):

    def test_native_round_trip(self) -> None:
        m = resnet_18()
        m.eval()
        x = lucid.randn(1, 3, 64, 64)
        out_before = m(x).last_hidden_state

        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = ResNet.from_pretrained(tmp)
            m2.eval()
            out_after = m2(x).last_hidden_state

        diff = float((out_before - out_after).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_safetensors_round_trip(self) -> None:
        m = resnet_18()
        m.eval()
        x = lucid.randn(1, 3, 64, 64)
        out_before = m(x).last_hidden_state

        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp, safe_serialization=True)
            m2 = ResNet.from_pretrained(tmp)
            m2.eval()
            out_after = m2(x).last_hidden_state

        diff = float((out_before - out_after).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_config_saved_as_json(self) -> None:
        import json, os
        m = resnet_50_cls()
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            with open(os.path.join(tmp, "config.json")) as f:
                d = json.load(f)
        self.assertEqual(d["model_type"], "resnet")
        self.assertEqual(d["block_type"], "bottleneck")
        self.assertEqual(d["layers"], [3, 4, 6, 3])

    def test_auto_class_loads_from_directory(self) -> None:
        m = resnet_50_cls()
        m.eval()
        x = lucid.randn(1, 3, 64, 64)
        logits_before = m(x).logits

        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp)
            m2 = models.AutoModelForImageClassification.from_pretrained(tmp)
            m2.eval()
            logits_after = m2(x).logits

        diff = float((logits_before - logits_after).abs().max().item())
        self.assertAlmostEqual(diff, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
