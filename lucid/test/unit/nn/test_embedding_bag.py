"""``F.embedding_bag`` correctness — guards the CPU-kernel int64-index + max-init fix."""

import numpy as np
import pytest

import lucid
from lucid.nn.functional import embedding_bag

ref = pytest.importorskip("torch")


def _ref(idx: np.ndarray, off: np.ndarray | None, W: np.ndarray, **kw: object) -> np.ndarray:
    toff = ref.tensor(off) if off is not None else None
    return ref.nn.functional.embedding_bag(
        ref.tensor(idx), ref.tensor(W), toff, **kw
    ).numpy()


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("idx_dtype", [np.int64, np.int32])
def test_1d_offsets_matches_reference(mode: str, idx_dtype: type) -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((12, 6)).astype(np.float32)
    idx = np.array([1, 2, 4, 5, 4, 3, 2, 0], dtype=idx_dtype)
    off = np.array([0, 3, 5], dtype=idx_dtype)
    got = embedding_bag(
        lucid.from_numpy(idx), lucid.from_numpy(W), offsets=lucid.from_numpy(off), mode=mode
    ).numpy()
    want = _ref(idx.astype(np.int64), off.astype(np.int64), W, mode=mode)
    assert np.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
def test_2d_matches_reference(mode: str) -> None:
    rng = np.random.default_rng(1)
    W = rng.standard_normal((12, 6)).astype(np.float32)
    idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    got = embedding_bag(lucid.from_numpy(idx), lucid.from_numpy(W), mode=mode).numpy()
    want = ref.nn.functional.embedding_bag(
        ref.tensor(idx), ref.tensor(W), mode=mode
    ).numpy()
    assert np.allclose(got, want, atol=1e-5)


def test_max_all_negative_bag() -> None:
    # The max-init bug: an all-negative bag was masked by the 0-seed → 0.
    W = -np.abs(np.random.default_rng(2).standard_normal((6, 4))).astype(np.float32)
    idx = np.array([0, 1, 2], dtype=np.int64)
    off = np.array([0], dtype=np.int64)
    got = embedding_bag(
        lucid.from_numpy(idx), lucid.from_numpy(W), offsets=lucid.from_numpy(off), mode="max"
    ).numpy()
    want = _ref(idx, off, W, mode="max")
    assert np.allclose(got, want, atol=1e-5)


def test_padding_idx() -> None:
    rng = np.random.default_rng(3)
    W = rng.standard_normal((10, 5)).astype(np.float32)
    idx = np.array([1, 2, 2, 3, 4], dtype=np.int64)
    off = np.array([0, 2], dtype=np.int64)
    got = embedding_bag(
        lucid.from_numpy(idx),
        lucid.from_numpy(W),
        offsets=lucid.from_numpy(off),
        mode="sum",
        padding_idx=2,
    ).numpy()
    want = _ref(idx, off, W, mode="sum", padding_idx=2)
    assert np.allclose(got, want, atol=1e-5)
