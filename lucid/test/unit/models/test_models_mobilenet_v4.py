"""Unit tests for MobileNet v4 (Qin et al., 2024)."""

import math
import unittest

import lucid
import lucid.models as models
import lucid.nn as nn
from lucid.models.vision.mobilenet_v4 import (
    MobileNetV4Config,
    MobileNetV4,
    MobileNetV4ForImageClassification,
    mobilenet_v4_conv_small,
    mobilenet_v4_conv_small_cls,
)

#: (factory stem, headless params, params with the classifier).  Exact
#: counts of the authors' released models; the paper's Table 6 rounds the
#: same numbers (3.8 / 9.2 / 31 / 10.5 / 35.9 M, the larger ones in 2**20
#: units).
_VARIANTS: tuple[tuple[str, int, int], ...] = (
    ("conv_small", 1_261_664, 3_774_024),
    ("conv_medium", 7_203_152, 9_715_512),
    ("conv_large", 30_078_504, 32_590_864),
    ("hybrid_medium", 8_562_288, 11_074_648),
    ("hybrid_large", 35_252_264, 37_764_624),
)


class TestMobileNetV4Config(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = MobileNetV4Config()
        self.assertEqual(cfg.model_type, "mobilenet_v4")

    def test_unknown_variant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MobileNetV4Config(variant="conv_tiny")

    def test_rates_validated(self) -> None:
        with self.assertRaises(ValueError):
            MobileNetV4Config(dropout=1.0)
        with self.assertRaises(ValueError):
            MobileNetV4Config(drop_path_rate=-0.1)


class TestMobileNetV4Backbone(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v4_conv_small()
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


class TestMobileNetV4Classifier(unittest.TestCase):

    def setUp(self) -> None:
        self.model = mobilenet_v4_conv_small_cls()
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
        m = MobileNetV4ForImageClassification(MobileNetV4Config(num_classes=10))
        m.eval()
        x = lucid.randn(1, 3, 224, 224)
        self.assertEqual(m(x).logits.shape, (1, 10))


class TestMobileNetV4Registry(unittest.TestCase):

    def test_variants_registered(self) -> None:
        names = models.list_models(family="mobilenet_v4")
        self.assertIn("mobilenet_v4_conv_small", names)
        self.assertIn("mobilenet_v4_conv_small_cls", names)

    def test_create_model(self) -> None:
        m = models.create_model("mobilenet_v4_conv_small")
        self.assertIsInstance(m, MobileNetV4)


class TestMobileNetV4Variants(unittest.TestCase):

    def test_param_counts_match_reference(self) -> None:
        for name, headless, full in _VARIANTS:
            with self.subTest(variant=name):
                backbone = models.create_model(f"mobilenet_v4_{name}")
                classifier = models.create_model(f"mobilenet_v4_{name}_cls")
                self.assertEqual(backbone.num_parameters(), headless)
                self.assertEqual(classifier.num_parameters(), full)

    def test_hybrid_forward(self) -> None:
        # Mobile MQA reduces keys/values by a stride-2 depthwise conv; an
        # input whose stride-16 map is odd (7x7 at 112) must still work.
        model = models.create_model("mobilenet_v4_hybrid_medium_cls").eval()
        out = model(lucid.randn(1, 3, 112, 112))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_feature_info_matches_forward_features(self) -> None:
        model = mobilenet_v4_conv_small().eval()
        feat = model.forward_features(lucid.randn(1, 3, 224, 224))
        last = model.feature_info[-1]
        self.assertEqual(feat.shape[1], last.num_channels)
        self.assertEqual(224 // feat.shape[-1], last.reduction)

    def test_backbone_pretrained_refused(self) -> None:
        # The checkpoints carry the classifier head; the headless backbone
        # has none of its own and must fail loudly, never hand back
        # random weights.
        with self.assertRaises(NotImplementedError):
            mobilenet_v4_conv_small(pretrained=True)


class TestMobileNetV4Init(unittest.TestCase):
    """The released implementation's initialisation, measured."""

    def test_depthwise_fan_out_divides_by_groups(self) -> None:
        # A depthwise k x k kernel has fan-out k^2 in the reference
        # convention, so std = sqrt(2 / 25) for the 5x5 below — far wider
        # than a fan-out of k^2 * C would give.
        model = models.create_model("mobilenet_v4_conv_large_cls")
        dw = model.blocks[3][0].dw_mid.conv
        self.assertEqual(dw.groups, dw.out_channels)
        std = float(dw.weight.std().item())
        self.assertAlmostEqual(std, math.sqrt(2.0 / 25.0), delta=0.05 * std)

    def test_pointwise_fan_out(self) -> None:
        model = models.create_model("mobilenet_v4_conv_large_cls")
        pw = model.blocks[3][0].pw_exp.conv
        std = float(pw.weight.std().item())
        expected = math.sqrt(2.0 / pw.out_channels)
        self.assertAlmostEqual(std, expected, delta=0.05 * expected)

    def test_hybrid_layer_scale_starts_small(self) -> None:
        model = models.create_model("mobilenet_v4_hybrid_medium_cls")
        gammas = [m.gamma for _, m in model.named_modules() if hasattr(m, "gamma")]
        self.assertGreater(len(gammas), 0)
        for g in gammas:
            self.assertAlmostEqual(float(g.abs().max().item()), 1e-5, places=9)

    def test_conv_variants_have_no_layer_scale(self) -> None:
        model = mobilenet_v4_conv_small_cls()
        self.assertFalse(any(hasattr(m, "gamma") for m in model.modules()))

    def test_classifier_uniform_fan_out(self) -> None:
        model = mobilenet_v4_conv_small_cls()
        w = model.classifier.weight
        bound = 1.0 / math.sqrt(1000)
        self.assertLessEqual(float(w.abs().max().item()), bound)
        self.assertIsInstance(model.classifier, nn.Linear)


#: (factory stem, enum name, tag, timm source, crop, resize, acc@1, acc@5).
#: Accuracies are timm's ``results-imagenet.csv`` rows for each exact tag
#: at its train resolution — the resolution the preset reproduces.
_WEIGHTS: tuple[tuple[str, str, str, str, int, int, float, float], ...] = (
    (
        "conv_small",
        "MobileNetV4ConvSmallWeights",
        "E2400_R224_IN1K",
        "timm/mobilenetv4_conv_small.e2400_r224_in1k",
        224,
        256,
        73.756,
        91.430,
    ),
    (
        "conv_medium",
        "MobileNetV4ConvMediumWeights",
        "E500_R256_IN1K",
        "timm/mobilenetv4_conv_medium.e500_r256_in1k",
        256,
        269,
        79.916,
        95.188,
    ),
    (
        "conv_large",
        "MobileNetV4ConvLargeWeights",
        "E600_R384_IN1K",
        "timm/mobilenetv4_conv_large.e600_r384_in1k",
        384,
        404,
        82.974,
        96.244,
    ),
    (
        "hybrid_medium",
        "MobileNetV4HybridMediumWeights",
        "IX_E550_R256_IN1K",
        "timm/mobilenetv4_hybrid_medium.ix_e550_r256_in1k",
        256,
        269,
        81.478,
        95.692,
    ),
    (
        "hybrid_large",
        "MobileNetV4HybridLargeWeights",
        "IX_E600_R384_IN1K",
        "timm/mobilenetv4_hybrid_large.ix_e600_r384_in1k",
        384,
        404,
        83.996,
        96.714,
    ),
)


class TestMobileNetV4WeightsEnums(unittest.TestCase):
    """Static contract of the per-variant Weights enums — no network."""

    def test_default_alias(self) -> None:
        import lucid.models.weights as W

        for _, enum_name, tag, *_ in _WEIGHTS:
            with self.subTest(enum=enum_name):
                cls = getattr(W, enum_name)
                self.assertIs(cls.DEFAULT, getattr(cls, tag))
                self.assertEqual(list(cls.__members__), [tag, "DEFAULT"])

    def test_entry_fields(self) -> None:
        import lucid.models.weights as W

        for stem, enum_name, tag, source, _, _, top1, top5 in _WEIGHTS:
            with self.subTest(enum=enum_name):
                member = getattr(getattr(W, enum_name), tag)
                e = member.entry
                self.assertEqual(e.num_classes, 1000)
                self.assertEqual(len(e.sha256), 64)
                self.assertIn(f"lucid-dl/mobilenet-v4-{stem.replace('_', '-')}/", e.url)
                self.assertIn(f"/{tag}/", e.url)
                meta = member.meta
                self.assertEqual(meta["tag"], tag)
                self.assertEqual(meta["source"], source)
                self.assertEqual(meta["license"], "apache-2.0")
                classifier = models.create_model(f"mobilenet_v4_{stem}_cls")
                self.assertEqual(meta["num_params"], classifier.num_parameters())
                acc = meta["metrics"]["ImageNet-1k"]
                self.assertAlmostEqual(acc["acc@1"], top1)
                self.assertAlmostEqual(acc["acc@5"], top5)

    def test_transforms_match_source_pipeline(self) -> None:
        import lucid.models.weights as W

        for _, enum_name, tag, _, crop, resize, *_ in _WEIGHTS:
            with self.subTest(enum=enum_name):
                tf = getattr(getattr(W, enum_name), tag).transforms()
                self.assertEqual(tf.crop_size, crop)
                self.assertEqual(tf.resize_size, resize)
                self.assertEqual(tf.interpolation, "bicubic")
                self.assertEqual(tuple(tf.mean), (0.485, 0.456, 0.406))
                self.assertEqual(tuple(tf.std), (0.229, 0.224, 0.225))

    def test_registry_discoverable(self) -> None:
        from lucid.weights import list_pretrained

        for stem, _, tag, *_ in _WEIGHTS:
            with self.subTest(variant=stem):
                self.assertIn(tag, list_pretrained(f"mobilenet_v4_{stem}_cls"))

    def test_family_package_reexports(self) -> None:
        import lucid.models.vision.mobilenet_v4 as pkg
        import lucid.models.weights as W

        for _, enum_name, *_ in _WEIGHTS:
            with self.subTest(enum=enum_name):
                self.assertIs(getattr(pkg, enum_name), getattr(W, enum_name))


@unittest.skipUnless(
    __import__("os").environ.get("LUCID_TEST_NETWORK") == "1",
    "set LUCID_TEST_NETWORK=1 to exercise the Hugging Face Hub download",
)
class TestMobileNetV4PretrainedLoad(unittest.TestCase):
    """End-to-end: download + SHA-verify + load into model."""

    def test_conv_small_default(self) -> None:
        m = mobilenet_v4_conv_small_cls(pretrained=True)
        m.eval()
        out = m(lucid.randn(1, 3, 224, 224))
        self.assertEqual(out.logits.shape, (1, 1000))

    def test_hybrid_medium_string_tag(self) -> None:
        m = models.mobilenet_v4_hybrid_medium_cls(pretrained="IX_E550_R256_IN1K")
        self.assertIsInstance(m, MobileNetV4ForImageClassification)
        out = m.eval()(lucid.randn(1, 3, 256, 256))
        self.assertEqual(out.logits.shape, (1, 1000))


if __name__ == "__main__":
    unittest.main()
