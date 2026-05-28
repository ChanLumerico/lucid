"""Preset round-trip + multi-target contract tests.

Each of the 5 registered presets is exercised on three axes:

1. :meth:`to_dict` / :meth:`from_dict` is a fixed-point — repeat
   round-trips do not drift.
2. :class:`AutoTransformsPreset.from_dict` resolves the right
   subclass.
3. The preset, called on a multi-target sample dict (image + mask /
   image + boxes / image + keypoints), threads each target through
   the inner geometric stages with consistent params so coords /
   labels stay aligned.

These are the contract the pretrained-weights system relies on.
"""

import pytest

import lucid
import lucid.utils.transforms as T

# ── round-trip ──────────────────────────────────────────────────────


def _round_trip(preset: T.TransformsPreset) -> None:
    cfg = preset.to_dict()
    back = T.AutoTransformsPreset.from_dict(cfg)
    assert type(back) is type(
        preset
    ), f"{type(preset).__name__}: AutoTransformsPreset returned {type(back).__name__}"
    assert (
        back.to_dict() == cfg
    ), f"{type(preset).__name__}: round-trip drifted\n  before={cfg}\n  after={back.to_dict()}"
    # Fixed-point: a second round-trip must equal the first.
    cfg2 = back.to_dict()
    back2 = T.AutoTransformsPreset.from_dict(cfg2)
    assert back2.to_dict() == cfg2


class TestRoundTrip:
    def test_image_classification(self) -> None:
        _round_trip(T.ImageClassification(crop_size=224, resize_size=256))

    def test_image_classification_custom_stats(self) -> None:
        _round_trip(
            T.ImageClassification(
                crop_size=299,
                resize_size=342,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                interpolation="bicubic",
            )
        )

    def test_image_classification_augment(self) -> None:
        _round_trip(T.ImageClassificationAugment(crop_size=224))

    def test_image_classification_augment_disabled_aux(self) -> None:
        # color_jitter=0 + hflip_prob=0 should still round-trip cleanly.
        _round_trip(
            T.ImageClassificationAugment(
                crop_size=224, color_jitter=0.0, hflip_prob=0.0
            )
        )

    def test_detection(self) -> None:
        _round_trip(T.Detection(max_size=1333, min_area=4.0, min_visibility=0.3))

    def test_segmentation(self) -> None:
        _round_trip(T.Segmentation(crop_size=480, resize_size=520))

    def test_pose(self) -> None:
        _round_trip(T.Pose(crop_size=192, resize_size=224))


# ── registry / resolver ─────────────────────────────────────────────


class TestAutoResolver:
    def test_registered_names(self) -> None:
        names = T.AutoTransformsPreset.registered()
        assert names == [
            "Detection",
            "ImageClassification",
            "ImageClassificationAugment",
            "Pose",
            "Segmentation",
        ]

    def test_unknown_preset_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            T.AutoTransformsPreset.from_dict(
                {"preprocessor_type": "DoesNotExist", "init_kwargs": {}}
            )

    def test_missing_preprocessor_type_raises(self) -> None:
        with pytest.raises(ValueError):
            T.AutoTransformsPreset.from_dict({"init_kwargs": {}})

    def test_non_dict_init_kwargs_raises(self) -> None:
        with pytest.raises(ValueError):
            T.AutoTransformsPreset.from_dict(
                {"preprocessor_type": "ImageClassification", "init_kwargs": []}
            )

    def test_concrete_class_from_dict_validates_type(self) -> None:
        # ImageClassification.from_dict on a Detection cfg must raise.
        cfg = T.Detection().to_dict()
        with pytest.raises(ValueError):
            T.ImageClassification.from_dict(cfg)


# ── multi-target threading ──────────────────────────────────────────


class TestMultiTarget:
    def test_classification_passes_lone_tensor(self) -> None:
        # ImageClassification is image-only; passing a bare tensor
        # returns a bare tensor (no dict wrapping).
        img = lucid.rand(3, 300, 400)
        out = T.ImageClassification(crop_size=224)(img)
        assert tuple(out.shape) == (3, 224, 224)  # type: ignore[union-attr]

    def test_segmentation_image_and_mask(self) -> None:
        # Mask must follow the geometric chain — same final HxW, no
        # interpolation-induced label leakage.
        lucid.manual_seed(0)
        img = lucid.rand(3, 300, 400)
        # 5-label mask
        m = lucid.floor(lucid.rand(1, 300, 400) * 5.0)
        sample = {"image": T.Image(img), "mask": T.Mask(m)}
        out = T.Segmentation(crop_size=224, resize_size=256)(sample)
        assert tuple(out["image"].data.shape) == (3, 224, 224)
        assert tuple(out["mask"].data.shape) == (1, 224, 224)
        # No synthetic labels (nearest interp only).
        before = set(int(round(v)) for v in m.numpy().reshape(-1).tolist())
        after = set(
            int(round(v)) for v in out["mask"].data.numpy().reshape(-1).tolist()
        )
        assert after <= before, f"mask gained labels {after - before}"

    def test_detection_image_and_boxes(self) -> None:
        # Boxes must update canvas + drop nothing in this safe case.
        img = lucid.rand(3, 400, 600)
        boxes = T.BoundingBoxes(
            lucid.tensor([[50.0, 50.0, 250.0, 250.0], [300.0, 200.0, 500.0, 380.0]]),
            "xyxy",
            (400, 600),
            labels=lucid.tensor([1.0, 2.0]),
        )
        sample = {"image": T.Image(img), "boxes": boxes}
        out = T.Detection(max_size=512, min_area=1.0)(sample)
        # All boxes survive (both well-inside the canvas before scaling).
        assert int(out["boxes"].data.shape[0]) == 2
        assert out["boxes"].labels.numpy().tolist() == [1.0, 2.0]

    def test_detection_drops_tiny_box(self) -> None:
        # A 1-pixel box should fall below min_area=10.
        img = lucid.rand(3, 256, 256)
        boxes = T.BoundingBoxes(
            lucid.tensor([[100.0, 100.0, 101.0, 101.0]]),
            "xyxy",
            (256, 256),
            labels=lucid.tensor([7.0]),
        )
        # Make pipeline a no-op size-wise then assert the filter
        # kicks the 1-pixel box.
        out = T.Detection(max_size=256, min_area=10.0)(
            {"image": T.Image(img), "boxes": boxes}
        )
        assert int(out["boxes"].data.shape[0]) == 0

    def test_pose_image_and_keypoints(self) -> None:
        img = lucid.rand(3, 400, 600)
        kps = T.Keypoints(
            lucid.tensor([[100.0, 50.0, 1.0], [300.0, 200.0, 1.0]]),
            canvas_size=(400, 600),
        )
        sample = {"image": T.Image(img), "kps": kps}
        out = T.Pose(crop_size=224, resize_size=256)(sample)
        # Count + visibility column preserved.
        assert int(out["kps"].data.shape[0]) == 2
        assert int(out["kps"].data.shape[1]) == 3
        extras = out["kps"].data.numpy()[:, 2].tolist()
        assert all(abs(v - 1.0) < 1e-5 for v in extras)


class TestImageClassificationAugmentStrong:
    """Phase 8 — `ImageClassificationAugment` extended kwargs.

    Verifies the new `auto_augment` and `random_erasing` knobs:
    pipeline assembly, round-trip, runtime correctness.
    """

    def test_no_optional_stages_by_default(self) -> None:
        # The default is the baseline recipe — no AutoAugment, no RandomErasing.
        preset = T.ImageClassificationAugment(crop_size=224)
        x = lucid.rand(3, 256, 256)
        y = preset(x)
        assert tuple(y.shape) == (3, 224, 224)

    def test_auto_augment_ta_wide_runs(self) -> None:
        preset = T.ImageClassificationAugment(
            crop_size=64, auto_augment="ta_wide"
        )
        y = preset(lucid.rand(3, 96, 96))
        assert tuple(y.shape) == (3, 64, 64)

    def test_auto_augment_ra_with_params(self) -> None:
        preset = T.ImageClassificationAugment(
            crop_size=64, auto_augment="ra-m9-n2"
        )
        y = preset(lucid.rand(3, 96, 96))
        assert tuple(y.shape) == (3, 64, 64)

    def test_auto_augment_aa_imagenet(self) -> None:
        preset = T.ImageClassificationAugment(
            crop_size=64, auto_augment="aa_imagenet"
        )
        y = preset(lucid.rand(3, 96, 96))
        assert tuple(y.shape) == (3, 64, 64)

    def test_random_erasing_changes_output(self) -> None:
        # With random_erasing=1.0 the erase fires every call.
        lucid.manual_seed(0)
        preset = T.ImageClassificationAugment(
            crop_size=64,
            random_erasing=1.0,
            color_jitter=0.0,
            hflip_prob=0.0,
        )
        y = preset(lucid.ones(3, 64, 64))
        # Some pixels should be replaced (the erase region != input).
        assert float(y.min().item()) != float(y.max().item())

    def test_unknown_auto_augment_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown auto_augment"):
            T.ImageClassificationAugment(crop_size=224, auto_augment="bogus")

    def test_invalid_random_erasing_raises(self) -> None:
        with pytest.raises(ValueError, match="random_erasing"):
            T.ImageClassificationAugment(crop_size=224, random_erasing=1.5)

    def test_round_trip_with_strong_aug(self) -> None:
        # to_dict / from_dict must preserve the new kwargs.
        preset = T.ImageClassificationAugment(
            crop_size=224,
            auto_augment="ra-m9-n2",
            random_erasing=0.25,
        )
        cfg = preset.to_dict()
        assert cfg["init_kwargs"]["auto_augment"] == "ra-m9-n2"
        assert cfg["init_kwargs"]["random_erasing"] == 0.25
        restored = T.AutoTransformsPreset.from_dict(cfg)
        assert isinstance(restored, T.ImageClassificationAugment)
        assert restored.to_dict() == cfg

    def test_round_trip_defaults_yield_neutral_kwargs(self) -> None:
        # Default kwargs (no AutoAugment, no RandomErasing) survive round-trip.
        preset = T.ImageClassificationAugment(crop_size=224)
        cfg = preset.to_dict()
        assert cfg["init_kwargs"]["auto_augment"] is None
        assert cfg["init_kwargs"]["random_erasing"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
