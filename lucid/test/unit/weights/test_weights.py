"""Unit tests for the ``lucid.weights`` pretrained-weight system.

Covers the runtime surface that does not require network access:
  * WeightsEnum structure + DEFAULT aliasing + member accessors
  * Discovery: list_pretrained / get_weight
  * resolve_weights selection logic (bool / str / enum / errors)
  * ImageClassification transform shape + normalization behavior
  * Factory wiring: resnet_18_cls(pretrained=...) selects/loads correctly
    (random path; the download path is exercised by the parity suite once
    real checkpoints are uploaded)
"""

import pytest

import lucid
import lucid.weights as W
from lucid.utils.transforms import ImageClassification
from lucid.weights import WeightEntry, WeightsEnum
from lucid.models.vision.resnet import ResNet18Weights


# ── WeightsEnum structure ───────────────────────────────────────────


class TestWeightsEnum:
    def test_default_aliases_canonical(self) -> None:
        assert ResNet18Weights.DEFAULT is ResNet18Weights.IMAGENET1K_V1

    def test_member_accessors(self) -> None:
        w = ResNet18Weights.IMAGENET1K_V1
        assert w.tag == "IMAGENET1K_V1"
        assert w.num_classes == 1000
        assert w.url.endswith("/IMAGENET1K_V1/model.safetensors")
        assert isinstance(w.entry, WeightEntry)

    def test_meta_present(self) -> None:
        meta = ResNet18Weights.IMAGENET1K_V1.meta
        assert meta["source"].startswith("torchvision/")
        assert meta["license"] == "bsd-3-clause"
        assert meta["metrics"]["ImageNet-1k"]["acc@1"] == pytest.approx(69.758)

    def test_transforms_returns_callable(self) -> None:
        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        assert isinstance(tf, ImageClassification)

    def test_iteration_skips_default_alias(self) -> None:
        # Enum iteration yields canonical members only.
        names = [m.name for m in ResNet18Weights]
        assert "IMAGENET1K_V1" in names
        assert "DEFAULT" not in names


# ── Discovery ───────────────────────────────────────────────────────


class TestDiscovery:
    def test_list_pretrained(self) -> None:
        assert W.list_pretrained("resnet_18") == ["IMAGENET1K_V1"]

    def test_list_pretrained_unknown(self) -> None:
        assert W.list_pretrained("does_not_exist") == []

    def test_get_weight(self) -> None:
        w = W.get_weight("ResNet18Weights.IMAGENET1K_V1")
        assert w is ResNet18Weights.IMAGENET1K_V1

    def test_get_weight_default_alias(self) -> None:
        w = W.get_weight("ResNet18Weights.DEFAULT")
        assert w is ResNet18Weights.IMAGENET1K_V1

    def test_get_weight_bad_format(self) -> None:
        with pytest.raises(ValueError, match="EnumName.TAG"):
            W.get_weight("ResNet18Weights")

    def test_get_weight_unknown_enum(self) -> None:
        with pytest.raises(ValueError, match="unknown weights enum"):
            W.get_weight("NopeWeights.TAG")

    def test_get_weight_unknown_tag(self) -> None:
        with pytest.raises(ValueError, match="no tag"):
            W.get_weight("ResNet18Weights.IMAGENET1K_V99")


# ── resolve_weights ─────────────────────────────────────────────────


class TestResolveWeights:
    def test_false_returns_none(self) -> None:
        assert W.resolve_weights(ResNet18Weights, False, None) is None

    def test_true_returns_default(self) -> None:
        assert (
            W.resolve_weights(ResNet18Weights, True, None)
            is ResNet18Weights.IMAGENET1K_V1
        )

    def test_string_tag(self) -> None:
        assert (
            W.resolve_weights(ResNet18Weights, "IMAGENET1K_V1", None)
            is ResNet18Weights.IMAGENET1K_V1
        )

    def test_explicit_weights_wins(self) -> None:
        # weights= takes precedence over pretrained.
        out = W.resolve_weights(
            ResNet18Weights, False, ResNet18Weights.IMAGENET1K_V1
        )
        assert out is ResNet18Weights.IMAGENET1K_V1

    def test_unknown_string_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="no tag"):
            W.resolve_weights(ResNet18Weights, "NOPE", None)

    def test_wrong_enum_member_raises(self) -> None:
        class OtherWeights(WeightsEnum):
            X = WeightEntry(
                url="http://x", sha256="", num_classes=1,
                transforms=ImageClassification(crop_size=1),
            )

        with pytest.raises(TypeError, match="not a member"):
            W.resolve_weights(ResNet18Weights, False, OtherWeights.X)

    def test_bad_pretrained_type_raises(self) -> None:
        with pytest.raises(TypeError, match="bool or str"):
            W.resolve_weights(ResNet18Weights, 3.5, None)  # type: ignore[arg-type]


# ── ImageClassification transform ───────────────────────────────────


class TestImageClassification:
    def test_unbatched_shape(self) -> None:
        tf = ImageClassification(crop_size=224, resize_size=256)
        out = tf(lucid.rand(3, 300, 400))
        assert tuple(out.shape) == (3, 224, 224)

    def test_batched_shape(self) -> None:
        tf = ImageClassification(crop_size=224, resize_size=256)
        out = tf(lucid.rand(2, 3, 300, 400))
        assert tuple(out.shape) == (2, 3, 224, 224)

    def test_normalization_applied(self) -> None:
        # A constant image equal to the mean normalizes to ~0.
        tf = ImageClassification(crop_size=4, resize_size=4, mean=(0.5,), std=(0.5,))
        x = lucid.ones(1, 8, 8) * 0.5
        out = tf(x)
        assert abs(float(out.mean().item())) < 1e-5

    def test_square_crop(self) -> None:
        tf = ImageClassification(crop_size=128, resize_size=160)
        out = tf(lucid.rand(3, 200, 200))
        assert tuple(out.shape) == (3, 128, 128)


# ── Factory wiring ──────────────────────────────────────────────────


class TestFactoryWiring:
    def test_random_init_default(self) -> None:
        from lucid.models.vision.resnet import resnet_18_cls

        model = resnet_18_cls()  # pretrained=False
        assert type(model).__name__ == "ResNetForImageClassification"

    def test_pretrained_false_no_download(self) -> None:
        from lucid.models.vision.resnet import resnet_18_cls

        # Must not attempt any network access when pretrained is falsy.
        model = resnet_18_cls(pretrained=False)
        assert model is not None

    def test_overrides_still_apply(self) -> None:
        from lucid.models.vision.resnet import resnet_18_cls

        model = resnet_18_cls(num_classes=10)
        assert model.config.num_classes == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
