"""Parity for exact axis-aligned spatial ops (Transpose / Flip /
RandomRotate90 / RandomScale / D4 / RandomGridShuffle) vs Albumentations.

All ops are pure index reorderings (or near-exact resize at scale=1) —
bit-exact parity expected vs np.rot90 / np.flip / Albumentations.

The Lucid sampling RNG is pinned per test where needed so that random
ops produce a deterministic param choice we can mirror with numpy.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
from lucid.utils.transforms import _random as _trf_random

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


def _seed_for(target: int, draw: object, max_seeds: int = 256) -> int:
    """Find a ``lucid.manual_seed`` value that makes ``draw()`` return ``target``.

    ``draw`` is a zero-arg lambda that performs the same RNG draw the
    transform's ``make_params`` does — typically
    ``lambda: _trf_random.randint(lo, hi)``.
    """
    for s in range(max_seeds):
        lucid.manual_seed(s)
        if draw() == target:  # type: ignore[operator]
            return s
    raise RuntimeError(f"no seed in [0, {max_seeds}) yields target={target}")


def _image(seed: int = 0, h: int = 24, w: int = 32) -> tuple[lucid.Tensor, np.ndarray]:
    """Matched (CHW Lucid tensor, HWC numpy array) image pair."""
    hwc = np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid(tf: T.Transform, chw: lucid.Tensor) -> np.ndarray:
    out = tf(T.Image(chw)).data.numpy()
    return np.transpose(out, (1, 2, 0))


def _run_albu(aug: object, hwc: np.ndarray) -> np.ndarray:
    return aug(image=hwc)["image"]  # type: ignore[operator]


# ── Transpose ───────────────────────────────────────────────────────


@pytest.mark.parity
class TestTranspose:
    def test_exact_vs_albu(self) -> None:
        chw, hwc = _image(0)
        got = _run_lucid(T.Transpose(p=1.0), chw)
        ref = _run_albu(A.Transpose(p=1.0), hwc)
        assert got.shape == ref.shape
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)

    def test_exact_vs_numpy(self) -> None:
        # Albumentations Transpose swaps H and W → numpy axes (1, 0, 2).
        chw, hwc = _image(7)
        got = _run_lucid(T.Transpose(p=1.0), chw)
        ref = np.transpose(hwc, (1, 0, 2))
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


# ── Flip (random axis) ──────────────────────────────────────────────


@pytest.mark.parity
class TestFlip:
    """``Flip`` samples ``code ∈ {-1, 0, 1}`` via ``_random.randint(0, 3) - 1``.

    We pin the Lucid sampling RNG so the chosen axis is deterministic,
    then compare against the matching numpy ground truth.
    """

    @pytest.mark.parametrize("code", [-1, 0, 1])
    def test_each_axis_vs_numpy(self, code: int) -> None:
        chw, hwc = _image(1)
        seed = _seed_for(code, lambda: _trf_random.randint(0, 3) - 1)
        lucid.manual_seed(seed)

        got = _run_lucid(T.Flip(p=1.0), chw)
        # Mirror Lucid's logic: hflip if code >= 0, vflip if code <= 0.
        ref = hwc
        if code >= 0:
            ref = ref[:, ::-1, :]  # horizontal flip — reverse W
        if code <= 0:
            ref = ref[::-1, :, :]  # vertical flip — reverse H
        ref = np.ascontiguousarray(ref)
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


# ── RandomRotate90 ──────────────────────────────────────────────────


@pytest.mark.parity
class TestRandomRotate90:
    """Pins the Lucid sampling RNG so ``k`` is deterministic, then compares
    against ``np.rot90`` on the HWC array (axes ``(0, 1)``)."""

    @pytest.mark.parametrize("k", [0, 1, 2, 3])
    def test_each_k_vs_numpy(self, k: int) -> None:
        chw, hwc = _image(2)
        seed = _seed_for(k, lambda: _trf_random.randint(0, 4))
        lucid.manual_seed(seed)

        got = _run_lucid(T.RandomRotate90(p=1.0), chw)
        # lucid.rot90 dims=(-2, -1) over CHW == np.rot90 axes=(0, 1) over HWC,
        # since both rotate the spatial plane in the same sense.
        ref = np.ascontiguousarray(np.rot90(hwc, k=k, axes=(0, 1)))
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


# ── RandomScale (identity scale) ────────────────────────────────────


@pytest.mark.parity
class TestRandomScaleIdentity:
    def test_scale_zero_passes_through(self) -> None:
        chw, hwc = _image(3)
        got = _run_lucid(T.RandomScale(scale_limit=(0.0, 0.0), p=1.0), chw)
        # f = 1.0 → no resize → identity (bilinear on integer grid noop).
        np.testing.assert_allclose(got, hwc, atol=1e-5)

    def test_scale_zero_shape_preserved(self) -> None:
        chw, _ = _image(4)
        got = _run_lucid(T.RandomScale(scale_limit=(0.0, 0.0), p=1.0), chw)
        assert got.shape == (24, 32, 3)


# ── D4 (8 dihedral symmetries) ──────────────────────────────────────


@pytest.mark.parity
class TestD4:
    """D4 enumerates ``rot90^k`` then optional hflip. Pin RNG so the
    sampled group element ``g ∈ {0..7}`` matches the parametrized index,
    then compare against the manual numpy composition."""

    @pytest.mark.parametrize("g", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_each_dihedral_index_vs_numpy(self, g: int) -> None:
        chw, hwc = _image(5)
        seed = _seed_for(g, lambda: _trf_random.randint(0, 8), max_seeds=512)
        lucid.manual_seed(seed)

        got = _run_lucid(T.D4(p=1.0), chw)
        k = g % 4
        flip = g >= 4
        ref = np.rot90(hwc, k=k, axes=(0, 1))
        if flip:
            ref = ref[:, ::-1, :]  # hflip → reverse W
        ref = np.ascontiguousarray(ref)
        np.testing.assert_allclose(got, ref, atol=0.0, rtol=0.0)


# ── RandomGridShuffle ───────────────────────────────────────────────


@pytest.mark.parity
class TestRandomGridShuffle:
    def test_shape_preserved(self) -> None:
        chw, _ = _image(6)
        out = _run_lucid(T.RandomGridShuffle(grid=(3, 3), p=1.0), chw)
        assert out.shape == (24, 32, 3)

    def test_output_is_permutation_of_input(self) -> None:
        # H=24, W=32 with grid=(3, 3): cell widths 32//3=10 (×3) = 30,
        # so the last column has width 12 and resize is invoked. Use a
        # grid that divides evenly to keep the permutation pixel-exact.
        chw, hwc = _image(6, h=24, w=30)
        out = _run_lucid(T.RandomGridShuffle(grid=(3, 3), p=1.0), chw)
        # Even grid → no resize → pure index permutation → sorted pixel
        # values must match exactly.
        np.testing.assert_allclose(
            np.sort(out.reshape(-1)), np.sort(hwc.reshape(-1)), atol=0.0, rtol=0.0
        )

    def test_sum_approx_preserved_uneven_grid(self) -> None:
        # With an uneven split, resize(nearest) on the boundary cells
        # introduces small mass changes — still ballpark-preserving.
        chw, hwc = _image(8, h=24, w=32)
        out = _run_lucid(T.RandomGridShuffle(grid=(3, 3), p=1.0), chw)
        assert abs(out.sum() - hwc.sum()) < 1e-2 * hwc.sum()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
