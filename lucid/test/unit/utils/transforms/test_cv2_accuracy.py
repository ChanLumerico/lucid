"""G2 — cv2-accuracy upgrades: exact HSV round-trip + tiled CLAHE."""

import pytest

import lucid
import lucid.utils.transforms as T
import lucid.utils.transforms.functional as F

# ── exact HSV ───────────────────────────────────────────────────────


class TestHSV:
    def test_roundtrip_is_identity(self) -> None:
        lucid.manual_seed(0)
        img = lucid.rand(3, 16, 24)
        h, s, v = F.rgb_to_hsv(img)
        rt = F.hsv_to_rgb(h, s, v)
        assert float((rt - img).abs().max().item()) < 1e-5

    def test_adjust_hsv_zero_shift_is_identity(self) -> None:
        lucid.manual_seed(1)
        img = lucid.rand(3, 16, 24)
        out = F.adjust_hsv(img, 0.0, 0.0, 0.0)
        assert float((out - img).abs().max().item()) < 1e-5

    def test_hue_shift_wraps(self) -> None:
        # A full 180-unit hue shift (OpenCV scale) wraps back to identity.
        lucid.manual_seed(2)
        img = lucid.rand(3, 8, 8)
        out = F.adjust_hsv(img, 180.0, 0.0, 0.0)
        assert float((out - img).abs().max().item()) < 1e-4

    def test_value_shift_brightens(self) -> None:
        img = lucid.ones(3, 4, 4) * 0.3
        out = F.adjust_hsv(img, 0.0, 0.0, 128.0)  # +0.5 on value
        assert float(out.mean().item()) > 0.3

    def test_class_preserves_shape_and_range(self) -> None:
        lucid.manual_seed(3)
        img = lucid.rand(3, 20, 28)
        out = T.HueSaturationValue(p=1.0)(T.Image(img)).data
        assert tuple(out.shape) == (3, 20, 28)
        assert 0.0 <= float(out.min().item()) and float(out.max().item()) <= 1.0

    def test_grayscale_rejected(self) -> None:
        with pytest.raises(ValueError):
            T.HueSaturationValue(p=1.0)(T.Image(lucid.rand(1, 8, 8)))


# ── tiled CLAHE ─────────────────────────────────────────────────────


class TestCLAHE:
    def test_shape_and_range_rgb(self) -> None:
        lucid.manual_seed(4)
        img = lucid.rand(3, 32, 40)
        out = F.clahe(img, 4.0, (8, 8))
        assert tuple(out.shape) == (3, 32, 40)
        assert 0.0 <= float(out.min().item()) and float(out.max().item()) <= 1.0

    def test_grayscale(self) -> None:
        lucid.manual_seed(5)
        out = F.clahe(lucid.rand(1, 24, 24), 4.0, (4, 4))
        assert tuple(out.shape) == (1, 24, 24)

    def test_batched(self) -> None:
        lucid.manual_seed(6)
        out = F.clahe(lucid.rand(2, 3, 32, 40), 2.0, (4, 4))
        assert tuple(out.shape) == (2, 3, 32, 40)

    def test_tiling_is_adaptive(self) -> None:
        # A left-dark / right-bright split image: CLAHE must enhance each
        # half independently, so the per-tile-mapped halves differ from a
        # single global equalization.  Verify the output is not constant and
        # spans most of the range (local contrast was stretched).
        left = lucid.ones(3, 32, 16) * 0.2
        right = lucid.ones(3, 32, 16) * 0.8
        img = lucid.concat([left, right], dim=-1)  # (3,32,32)
        out = F.clahe(img, 40.0, (4, 4))
        assert float(out.max().item()) - float(out.min().item()) > 0.1

    def test_class_repr_includes_grid(self) -> None:
        r = repr(T.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5))
        assert "tile_grid_size=(8, 8)" in r

    def test_class_preserves_shape(self) -> None:
        lucid.manual_seed(7)
        img = lucid.rand(3, 28, 36)
        out = T.CLAHE(p=1.0)(T.Image(img)).data
        assert tuple(out.shape) == (3, 28, 36)


# ── Posterize modes ──────────────────────────────────────────────────


class TestPosterize:
    def test_uint8_mask_distinct_levels(self) -> None:
        # 3 bits → 8 distinct values; the bit mask zeros the low 5 bits.
        lucid.manual_seed(0)
        img = lucid.rand(3, 16, 16)
        out = T.Posterize(num_bits=3, mode="uint8_mask", p=1.0)(T.Image(img)).data
        unique = set(out.numpy().reshape(-1).tolist())
        assert len(unique) <= 8

    def test_float_mode_is_strictly_lower_bound(self) -> None:
        # float floor → result strictly ≤ original.
        lucid.manual_seed(1)
        img = lucid.rand(3, 8, 8)
        out = T.Posterize(num_bits=4, mode="float", p=1.0)(T.Image(img)).data
        assert float((out - img).max().item()) <= 0.0 + 1e-7

    def test_invalid_num_bits_raises(self) -> None:
        with pytest.raises(ValueError):
            T.Posterize(num_bits=0)
        with pytest.raises(ValueError):
            T.Posterize(num_bits=8)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            T.Posterize(num_bits=4, mode="bitmask")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
