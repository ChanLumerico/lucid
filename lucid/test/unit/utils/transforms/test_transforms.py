"""Phase 1 — lucid.utils.transforms (inference substrate).

Covers Compose + the deterministic transforms (Resize / CenterCrop /
Normalize / Rescale), the functional API, and the ImageClassification
preset.  All tensor-native (no numpy / PIL).
"""

import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import functional as F


# ── functional ──────────────────────────────────────────────────────


class TestFunctional:
    def test_resize_shorter_side_int(self) -> None:
        out = F.resize(lucid.rand(3, 200, 400), 100)
        # shorter side (200) -> 100, aspect preserved (400 -> 200)
        assert tuple(out.shape) == (3, 100, 200)

    def test_resize_exact_hw(self) -> None:
        out = F.resize(lucid.rand(3, 200, 400), (64, 48))
        assert tuple(out.shape) == (3, 64, 48)

    def test_resize_batched(self) -> None:
        out = F.resize(lucid.rand(2, 3, 200, 400), 100)
        assert tuple(out.shape) == (2, 3, 100, 200)

    def test_center_crop(self) -> None:
        out = F.center_crop(lucid.rand(3, 256, 256), 224)
        assert tuple(out.shape) == (3, 224, 224)

    def test_crop(self) -> None:
        out = F.crop(lucid.rand(3, 10, 10), 2, 3, 4, 5)
        assert tuple(out.shape) == (3, 4, 5)

    def test_hflip_vflip_shape(self) -> None:
        x = lucid.rand(3, 8, 8)
        assert tuple(F.hflip(x).shape) == (3, 8, 8)
        assert tuple(F.vflip(x).shape) == (3, 8, 8)

    def test_hflip_involution(self) -> None:
        x = lucid.rand(1, 4, 4)
        back = F.hflip(F.hflip(x))
        assert float((back - x).abs().max().item()) < 1e-6

    def test_pad(self) -> None:
        assert tuple(F.pad(lucid.rand(3, 8, 8), 2).shape) == (3, 12, 12)
        assert tuple(F.pad(lucid.rand(3, 8, 8), (1, 2, 3, 4)).shape) == (3, 15, 11)

    def test_normalize(self) -> None:
        x = lucid.ones(3, 4, 4) * 0.5
        out = F.normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        assert abs(float(out.mean().item())) < 1e-6

    def test_rescale(self) -> None:
        out = F.rescale(lucid.ones(1, 2, 2) * 255.0)
        assert abs(float(out.max().item()) - 1.0) < 1e-6


# ── class transforms ────────────────────────────────────────────────


class TestClassTransforms:
    def test_resize(self) -> None:
        out = T.Resize(128)(lucid.rand(3, 256, 256))
        assert tuple(out.shape) == (3, 128, 128)

    def test_center_crop(self) -> None:
        out = T.CenterCrop(224)(lucid.rand(3, 256, 256))
        assert tuple(out.shape) == (3, 224, 224)

    def test_normalize(self) -> None:
        out = T.Normalize((0.5,), (0.5,))(lucid.ones(1, 4, 4) * 0.5)
        assert abs(float(out.mean().item())) < 1e-6

    def test_rescale(self) -> None:
        out = T.Rescale()(lucid.ones(1, 2, 2) * 255.0)
        assert abs(float(out.max().item()) - 1.0) < 1e-6

    def test_repr(self) -> None:
        assert "Resize" in repr(T.Resize(256))
        assert "CenterCrop" in repr(T.CenterCrop(224))


# ── Compose ─────────────────────────────────────────────────────────


class TestCompose:
    def test_chain(self) -> None:
        tf = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        out = tf(lucid.rand(3, 300, 400))
        assert tuple(out.shape) == (3, 224, 224)

    def test_repr_lists_inner(self) -> None:
        tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
        r = repr(tf)
        assert "Compose" in r and "Resize" in r and "CenterCrop" in r


# ── ImageClassification preset ──────────────────────────────────────


class TestImageClassificationPreset:
    def test_unbatched(self) -> None:
        out = T.ImageClassification(crop_size=224, resize_size=256)(
            lucid.rand(3, 300, 400)
        )
        assert tuple(out.shape) == (3, 224, 224)

    def test_batched(self) -> None:
        out = T.ImageClassification(crop_size=224, resize_size=256)(
            lucid.rand(2, 3, 300, 400)
        )
        assert tuple(out.shape) == (2, 3, 224, 224)

    def test_normalization(self) -> None:
        tf = T.ImageClassification(crop_size=4, resize_size=4, mean=(0.5,), std=(0.5,))
        out = tf(lucid.ones(1, 8, 8) * 0.5)
        assert abs(float(out.mean().item())) < 1e-5

    def test_default_imagenet_stats(self) -> None:
        tf = T.ImageClassification(crop_size=224)
        assert tf.mean == (0.485, 0.456, 0.406)
        assert tf.std == (0.229, 0.224, 0.225)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
