"""Numerical parity of ``lucid.utils.transforms`` vs Albumentations.

Opt-in tier (mirrors the reference-framework parity policy): the whole
module auto-skips unless both ``albumentations`` and ``cv2`` are
importable.  Install them with the parity extras to run::

    pip install albumentations
    pytest -m parity lucid/test/parity/transforms/

Lucid transforms operate on ``(C, H, W)`` float tensors in ``[0, 1]``;
Albumentations on ``(H, W, C)`` arrays.  Each test builds one random
image, runs both, and compares after a transpose.

Two tiers:

* **exact** — transforms whose math is unambiguous (flips, crops,
  resize, normalize, grayscale): agreement to float32 epsilon.
* **ballpark** — transforms where Albumentations routes through an
  8-bit / cv2 intermediate (HSV, solarize threshold boundary) while
  Lucid stays in float; compared with a loose tolerance only to catch
  gross divergence, not bit parity.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
import lucid.utils.transforms.functional as F

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


def _image(seed: int, h: int = 24, w: int = 32) -> tuple[lucid.Tensor, np.ndarray]:
    """A matched (Lucid CHW tensor, Albumentations HWC array) image pair."""
    hwc = np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid(tf: T.Transform, chw: lucid.Tensor) -> np.ndarray:
    out = tf(T.Image(chw)).data.numpy()
    return np.transpose(out, (1, 2, 0))


def _run_albu(aug: object, hwc: np.ndarray) -> np.ndarray:
    return aug(image=hwc)["image"]  # type: ignore[operator]


# ── exact tier ──────────────────────────────────────────────────────


class TestExact:
    @pytest.mark.parametrize(
        "lucid_tf, albu_tf",
        [
            (T.HorizontalFlip(p=1.0), A.HorizontalFlip(p=1.0)),
            (T.VerticalFlip(p=1.0), A.VerticalFlip(p=1.0)),
            (T.Transpose(p=1.0), A.Transpose(p=1.0)),
            (T.InvertImg(p=1.0), A.InvertImg(p=1.0)),
            (T.CenterCrop(16, 20, p=1.0), A.CenterCrop(16, 20, p=1.0)),
            (T.Crop(2, 3, 18, 23, p=1.0), A.Crop(2, 3, 18, 23, p=1.0)),
            (
                T.Resize(12, 16, interpolation="nearest", p=1.0),
                A.Resize(12, 16, interpolation=0, p=1.0),
            ),
            (
                T.LongestMaxSize(16, interpolation="nearest", p=1.0),
                A.LongestMaxSize(16, interpolation=0, p=1.0),
            ),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_exact(self, lucid_tf: T.Transform, albu_tf: object) -> None:
        chw, hwc = _image(0)
        got = _run_lucid(lucid_tf, chw)
        ref = _run_albu(albu_tf, hwc)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    def test_resize_bilinear(self) -> None:
        chw, hwc = _image(1)
        for hh, ww in [(12, 16), (40, 50)]:
            got = _run_lucid(T.Resize(hh, ww, p=1.0), chw)
            ref = _run_albu(A.Resize(hh, ww, interpolation=1, p=1.0), hwc)
            np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_smallest_max_size_bilinear(self) -> None:
        chw, hwc = _image(2)
        got = _run_lucid(T.SmallestMaxSize(16, p=1.0), chw)
        ref = _run_albu(A.SmallestMaxSize(16, interpolation=1, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_normalize(self) -> None:
        chw, hwc = _image(3)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        got = _run_lucid(T.Normalize(mean, std, max_pixel_value=1.0, p=1.0), chw)
        ref = _run_albu(
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0), hwc
        )
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_to_gray(self) -> None:
        chw, hwc = _image(4)
        got = _run_lucid(T.ToGray(p=1.0), chw)
        ref = _run_albu(A.ToGray(p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=1e-5)


# ── ballpark tier (Albumentations routes through 8-bit / cv2) ───────


class TestBallpark:
    def test_solarize_boundary(self) -> None:
        # Differences are confined to pixels near the threshold, where the
        # flipped value 1-x ≈ x, so the magnitude stays tiny.
        chw, hwc = _image(5)
        got = _run_lucid(T.Solarize(threshold=128, p=1.0), chw)
        ref = _run_albu(A.Solarize(threshold=0.5, p=1.0), hwc)
        np.testing.assert_allclose(got, ref, atol=1e-2)

    def test_hue_saturation_value(self) -> None:
        # Albumentations converts to uint8 + cv2 HSV; Lucid stays in float,
        # so agreement is to ~1/255 quantization, not float epsilon.
        chw, hwc = _image(6)
        got = np.transpose(F.adjust_hsv(chw, 20.0, 30.0, 10.0).numpy(), (1, 2, 0))
        ref = _run_albu(
            A.HueSaturationValue(
                hue_shift_limit=(20, 20),
                sat_shift_limit=(30, 30),
                val_shift_limit=(10, 10),
                p=1.0,
            ),
            hwc,
        )
        assert np.abs(got - ref).max() < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
