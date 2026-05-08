"""Indexing / scatter / gather ops."""

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close, assert_equal_int


class TestGather:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        idx = lucid.tensor([[0, 2], [1, 0]], dtype=lucid.int64, device=device)
        out = lucid.gather(x, idx, dim=1).numpy()
        np.testing.assert_array_equal(out, [[1.0, 3.0], [5.0, 4.0]])


class TestScatter:
    # ``scatter`` / ``scatter_add`` are CPU-only for now; the MLX
    # ``scatter`` shape contract differs from Lucid's adapter and
    # raises.  See P2-C engine notes; tracked as a follow-up.

    def test_scatter(self, device_cpu_only: str) -> None:
        base = lucid.zeros(3, 4, device=device_cpu_only)
        idx = lucid.tensor([[0, 1], [2, 1], [0, 2]],
                           dtype=lucid.int32, device=device_cpu_only)
        src = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                           device=device_cpu_only)
        out = lucid.scatter(base, dim=1, index=idx, src=src).numpy()
        expected = np.zeros((3, 4))
        for r in range(3):
            for c in range(2):
                col = [[0, 1], [2, 1], [0, 2]][r][c]
                val = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]][r][c]
                expected[r, col] = val
        np.testing.assert_array_equal(out, expected)

    def test_scatter_add(self, device_cpu_only: str) -> None:
        base = lucid.zeros(4, device=device_cpu_only)
        idx = lucid.tensor([0, 0, 2], dtype=lucid.int32, device=device_cpu_only)
        src = lucid.tensor([1.0, 2.0, 3.0], device=device_cpu_only)
        out = base.scatter_add(0, idx, src).numpy()
        np.testing.assert_array_equal(out, [3.0, 0.0, 3.0, 0.0])


class TestIndexSelect:
    def test_dim0(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        idx = lucid.tensor([0, 2], dtype=lucid.int64, device=device)
        out = lucid.index_select(x, 0, idx).numpy()
        np.testing.assert_array_equal(out, [[1.0, 2.0], [5.0, 6.0]])


class TestMaskedSelect:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        mask = lucid.tensor([True, False, True, False],
                            dtype=lucid.bool_, device=device)
        out = lucid.masked_select(x, mask).numpy()
        np.testing.assert_array_equal(out, [1.0, 3.0])


class TestMaskedFill:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        mask = lucid.tensor([True, False, True, False],
                            dtype=lucid.bool_, device=device)
        out = lucid.masked_fill(x, mask, -1.0).numpy()
        np.testing.assert_array_equal(out, [-1.0, 2.0, -1.0, 4.0])


class TestPut:
    # ``put`` / ``index_put`` go through the same scatter kernel that
    # diverges on Metal — pinned to CPU.

    def test_overwrite(self, device_cpu_only: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device_cpu_only)
        idx = lucid.tensor([0, 3], dtype=lucid.int64, device=device_cpu_only)
        src = lucid.tensor([10.0, 20.0], device=device_cpu_only)
        out = lucid.put(x, idx, src).numpy()
        np.testing.assert_array_equal(out, [[10.0, 2.0], [3.0, 20.0]])

    def test_accumulate(self, device_cpu_only: str) -> None:
        x = lucid.zeros(4, device=device_cpu_only)
        idx = lucid.tensor([0, 0, 1], dtype=lucid.int64, device=device_cpu_only)
        src = lucid.tensor([1.0, 2.0, 3.0], device=device_cpu_only)
        out = lucid.put(x, idx, src, accumulate=True).numpy()
        np.testing.assert_array_equal(out, [3.0, 3.0, 0.0, 0.0])


class TestIndexPut:
    def test_2d_overwrite(self, device_cpu_only: str) -> None:
        x = lucid.zeros(3, 4, device=device_cpu_only)
        out = lucid.index_put(
            x,
            (
                lucid.tensor([0, 1], dtype=lucid.int64, device=device_cpu_only),
                lucid.tensor([2, 3], dtype=lucid.int64, device=device_cpu_only),
            ),
            lucid.tensor([10.0, 20.0], device=device_cpu_only),
        ).numpy()
        expected = np.zeros((3, 4))
        expected[0, 2] = 10.0
        expected[1, 3] = 20.0
        np.testing.assert_array_equal(out, expected)

    def test_inplace(self, device_cpu_only: str) -> None:
        x = lucid.zeros(3, 4, device=device_cpu_only)
        ret = lucid.index_put_(
            x,
            (
                lucid.tensor([0], dtype=lucid.int64, device=device_cpu_only),
                lucid.tensor([1], dtype=lucid.int64, device=device_cpu_only),
            ),
            lucid.tensor([5.0], device=device_cpu_only),
        )
        assert ret is x
        assert x.numpy()[0, 1] == 5.0


class TestNonzero:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 0.0, 2.0, 0.0, 3.0], device=device)
        out = lucid.nonzero(t).numpy()
        # ``nonzero`` returns coords as a (n, ndim) tensor.
        np.testing.assert_array_equal(out.flatten(), [0, 2, 4])


class TestUnique:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 2.0, 3.0, 1.0], device=device)
        out = lucid.unique(t).numpy()
        np.testing.assert_array_equal(np.sort(out), [1.0, 2.0, 3.0])


class TestWhere:
    def test_basic(self, device: str) -> None:
        cond = lucid.tensor([True, False, True], dtype=lucid.bool_, device=device)
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([10.0, 20.0, 30.0], device=device)
        out = lucid.where(cond, a, b).numpy()
        np.testing.assert_array_equal(out, [1.0, 20.0, 3.0])


class TestTake:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        idx = lucid.tensor([0, 3], dtype=lucid.int64, device=device)
        # ``take`` indexes into the flattened view.
        out = lucid.take(x, idx).numpy()
        np.testing.assert_array_equal(out, [1.0, 4.0])
