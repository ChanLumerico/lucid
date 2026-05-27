"""Phase 2 — augmentation transforms.

Covers the randomized geometric + photometric transforms: shape
correctness, manual_seed reproducibility, probability extremes, and the
photometric functional helpers (brightness/contrast/saturation/hue).
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import functional as F


# ── geometric augmentations ─────────────────────────────────────────


class TestGeometricAug:
    def test_random_resized_crop_square(self) -> None:
        out = T.RandomResizedCrop(224)(lucid.rand(3, 64, 80))
        assert tuple(out.shape) == (3, 224, 224)

    def test_random_resized_crop_batched(self) -> None:
        out = T.RandomResizedCrop(96)(lucid.rand(2, 3, 200, 150))
        assert tuple(out.shape) == (2, 3, 96, 96)

    def test_random_crop_with_padding(self) -> None:
        out = T.RandomCrop(32, padding=4)(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    def test_random_crop_no_padding(self) -> None:
        out = T.RandomCrop(20)(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 20, 20)

    def test_hflip_p1_flips(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.RandomHorizontalFlip(p=1.0)(x)
        assert float((out - F.hflip(x)).abs().max().item()) < 1e-6

    def test_hflip_p0_identity(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.RandomHorizontalFlip(p=0.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_vflip_p1(self) -> None:
        x = lucid.rand(3, 8, 8)
        out = T.RandomVerticalFlip(p=1.0)(x)
        assert float((out - F.vflip(x)).abs().max().item()) < 1e-6

    def test_pad(self) -> None:
        assert tuple(T.Pad(4)(lucid.rand(3, 8, 8)).shape) == (3, 16, 16)


# ── photometric augmentations ───────────────────────────────────────


class TestColorJitter:
    def test_shape_preserved(self) -> None:
        out = T.ColorJitter(0.4, 0.4, 0.4, 0.1)(lucid.rand(3, 32, 32))
        assert tuple(out.shape) == (3, 32, 32)

    def test_zero_is_noop(self) -> None:
        x = lucid.rand(3, 16, 16)
        out = T.ColorJitter(0, 0, 0, 0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_output_in_range(self) -> None:
        out = T.ColorJitter(0.5, 0.5, 0.5, 0.2)(lucid.rand(3, 16, 16))
        assert float(out.min().item()) >= -1e-5
        assert float(out.max().item()) <= 1.0 + 1e-5

    def test_batched(self) -> None:
        out = T.ColorJitter(0.3, 0.3)(lucid.rand(2, 3, 16, 16))
        assert tuple(out.shape) == (2, 3, 16, 16)


class TestRandomErasing:
    def test_p1_erases(self) -> None:
        x = lucid.ones(3, 32, 32)
        out = T.RandomErasing(p=1.0, value=0.0)(x)
        # Some pixels became 0 (the erased box).
        assert float(out.min().item()) == 0.0

    def test_p0_identity(self) -> None:
        x = lucid.rand(3, 32, 32)
        out = T.RandomErasing(p=0.0)(x)
        assert float((out - x).abs().max().item()) < 1e-6

    def test_shape_batched(self) -> None:
        out = T.RandomErasing(p=1.0)(lucid.rand(2, 3, 32, 32))
        assert tuple(out.shape) == (2, 3, 32, 32)


# ── functional photometric ──────────────────────────────────────────


class TestPhotometricFunctional:
    def test_brightness_noop(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((F.adjust_brightness(x, 1.0) - x).abs().max().item()) < 1e-6

    def test_saturation_noop(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((F.adjust_saturation(x, 1.0) - x).abs().max().item()) < 1e-5

    def test_grayscale_is_constant_across_channels(self) -> None:
        g = F.rgb_to_grayscale(lucid.rand(3, 8, 8), keep_channels=True)
        ch0 = g[0:1]
        assert float((g - ch0).abs().max().item()) < 1e-6

    def test_hue_noop(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert float((F.adjust_hue(x, 0.0) - x).abs().max().item()) < 1e-6

    def test_hue_roundtrip(self) -> None:
        x = lucid.rand(3, 8, 8)
        back = F.adjust_hue(F.adjust_hue(x, 0.3), -0.3)
        assert float((back - x).abs().max().item()) < 1e-4


# ── reproducibility ─────────────────────────────────────────────────


class TestReproducibility:
    def test_manual_seed_reproducible(self) -> None:
        lucid.manual_seed(123)
        a = T.RandomResizedCrop(64)(lucid.rand(3, 128, 128))
        lucid.manual_seed(123)
        b = T.RandomResizedCrop(64)(lucid.rand(3, 128, 128))
        assert float((a - b).abs().max().item()) == 0.0

    def test_colorjitter_reproducible(self) -> None:
        lucid.manual_seed(5)
        a = T.ColorJitter(0.4, 0.4, 0.4, 0.1)(lucid.rand(3, 16, 16))
        lucid.manual_seed(5)
        b = T.ColorJitter(0.4, 0.4, 0.4, 0.1)(lucid.rand(3, 16, 16))
        assert float((a - b).abs().max().item()) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
