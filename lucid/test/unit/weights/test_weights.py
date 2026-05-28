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
        out = W.resolve_weights(ResNet18Weights, False, ResNet18Weights.IMAGENET1K_V1)
        assert out is ResNet18Weights.IMAGENET1K_V1

    def test_unknown_string_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="no tag"):
            W.resolve_weights(ResNet18Weights, "NOPE", None)

    def test_wrong_enum_member_raises(self) -> None:
        class OtherWeights(WeightsEnum):
            X = WeightEntry(
                url="http://x",
                sha256="",
                num_classes=1,
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


# ── Preset ↔ pretrained pipeline integration (G0) ───────────────────


class TestPresetIntegration:
    """Pins the contract between WeightEntry.transforms, the on-Hub
    config.json preprocessing block, and AutoTransformsPreset round-trip.

    The preset's to_dict() shape is what the conversion tool now
    emits into config.json; if that schema drifts here the published
    metadata silently becomes unreadable.
    """

    def test_resnet18_entry_transforms_is_preset(self) -> None:
        # WeightEntry.transforms must be a registered preset subclass
        # (not a raw Compose) so AutoTransformsPreset can round-trip it.
        from lucid.utils.transforms import TransformsPreset

        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        assert isinstance(tf, TransformsPreset)
        assert tf.preset_type == "ImageClassification"

    def test_resnet18_to_dict_matches_expected_schema(self) -> None:
        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        cfg = tf.to_dict()
        assert set(cfg) == {"preprocessor_type", "init_kwargs"}
        kw = cfg["init_kwargs"]
        assert isinstance(kw, dict)
        # Pinned ResNet-18 ImageNet hyperparams.
        assert kw["crop_size"] == 224
        assert kw["resize_size"] == 256
        assert kw["mean"] == [0.485, 0.456, 0.406]
        assert kw["std"] == [0.229, 0.224, 0.225]
        assert kw["interpolation"] == "bilinear"

    def test_resnet18_round_trip_via_auto_resolver(self) -> None:
        from lucid.utils.transforms import AutoTransformsPreset

        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        back = AutoTransformsPreset.from_dict(tf.to_dict())
        assert type(back) is type(tf)
        assert back.to_dict() == tf.to_dict()

    def test_resnet18_preset_threads_multitarget_sample(self) -> None:
        # An ImageClassification preset called on a {image, mask}
        # sample must thread both through the inner geometric stages
        # (Normalize touches only the image, mask survives Resize +
        # CenterCrop unchanged in label set).
        import lucid.utils.transforms as T

        lucid.manual_seed(0)
        img = lucid.rand(3, 300, 400)
        mask_raw = lucid.floor(lucid.rand(1, 300, 400) * 5.0)
        sample = {"image": T.Image(img), "mask": T.Mask(mask_raw)}
        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        out = tf(sample)
        assert tuple(out["image"].data.shape) == (3, 224, 224)
        assert tuple(out["mask"].data.shape) == (1, 224, 224)
        before = {int(round(v)) for v in mask_raw.numpy().reshape(-1).tolist()}
        after = {
            int(round(v)) for v in out["mask"].data.numpy().reshape(-1).tolist()
        }
        assert after <= before, (
            f"mask gained synthetic labels {sorted(after - before)} — "
            "ImageClassification leaked the geometric chain past nearest."
        )


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
