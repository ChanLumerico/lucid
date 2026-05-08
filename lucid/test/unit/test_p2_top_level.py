"""Unit tests for the P2-A top-level gap-closing additions.

Covers: ``relu``/``sigmoid`` aliases, ``count_nonzero``, ``randperm``,
``frexp``, ``tril_indices``/``triu_indices``/``combinations``,
``finfo``/``iinfo``, threading getters/setters, determinism aliases.
"""

import math

import numpy as np
import pytest

import lucid


class TestActivationAliases:
    def test_relu_top_level(self) -> None:
        out = lucid.relu(lucid.tensor([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(out.numpy(), [0.0, 0.0, 1.0])

    def test_sigmoid_top_level(self) -> None:
        out = lucid.sigmoid(lucid.tensor([0.0])).item()
        assert abs(out - 0.5) < 1e-6

    def test_tanh_still_works(self) -> None:
        # Sanity — keep the existing tanh top-level alias working.
        out = lucid.tanh(lucid.tensor([0.0])).item()
        assert abs(out - 0.0) < 1e-6


class TestCountNonzero:
    def test_full_reduction(self) -> None:
        assert int(lucid.count_nonzero(lucid.tensor([0.0, 1.0, 0.0, 2.0])).item()) == 2

    def test_along_dim(self) -> None:
        out = lucid.count_nonzero(lucid.tensor([[1.0, 0.0], [0.0, 2.0]]), 0)
        np.testing.assert_array_equal(out.numpy(), [1, 1])

    def test_dtype_is_int64(self) -> None:
        out = lucid.count_nonzero(lucid.tensor([1.0, 2.0]))
        assert out.dtype is lucid.int64


class TestRandperm:
    def test_is_permutation(self) -> None:
        lucid.manual_seed(0)
        out = lucid.randperm(8).numpy()
        assert sorted(out.tolist()) == list(range(8))

    def test_zero_size(self) -> None:
        out = lucid.randperm(0)
        assert out.shape == (0,)
        assert out.dtype is lucid.int64

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            lucid.randperm(-1)


class TestFrexp:
    def test_powers_of_two(self) -> None:
        m, e = lucid.frexp(lucid.tensor([0.5, 1.0, 4.0]))
        # 0.5 = 0.5 * 2^0; 1.0 = 0.5 * 2^1; 4.0 = 0.5 * 2^3
        np.testing.assert_allclose(m.numpy(), [0.5, 0.5, 0.5], atol=1e-6)
        np.testing.assert_array_equal(e.numpy(), [0, 1, 3])

    def test_zero(self) -> None:
        m, e = lucid.frexp(lucid.tensor([0.0]))
        assert m.item() == 0.0 and int(e.item()) == 0

    def test_exponent_is_int32(self) -> None:
        _, e = lucid.frexp(lucid.tensor([1.0]))
        assert e.dtype is lucid.int32


class TestTriIndices:
    def test_tril_indices_3x3_main_diag(self) -> None:
        out = lucid.tril_indices(3).numpy()
        # Lower triangle including diagonal: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
        assert out.shape == (2, 6)
        rows, cols = out.tolist()
        assert list(zip(rows, cols)) == [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

    def test_triu_indices_3x3_main_diag(self) -> None:
        out = lucid.triu_indices(3).numpy()
        rows, cols = out.tolist()
        assert list(zip(rows, cols)) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    def test_tril_offset_negative(self) -> None:
        out = lucid.tril_indices(3, offset=-1).numpy()
        rows, cols = out.tolist()
        assert list(zip(rows, cols)) == [(1, 0), (2, 0), (2, 1)]


class TestCombinations:
    def test_pairs(self) -> None:
        out = lucid.combinations(lucid.tensor([1.0, 2.0, 3.0]), r=2).numpy()
        np.testing.assert_array_equal(out, [[1, 2], [1, 3], [2, 3]])

    def test_with_replacement(self) -> None:
        out = lucid.combinations(
            lucid.tensor([1.0, 2.0]), r=2, with_replacement=True
        ).numpy()
        np.testing.assert_array_equal(out, [[1, 1], [1, 2], [2, 2]])

    def test_empty(self) -> None:
        out = lucid.combinations(lucid.tensor([1.0]), r=2)
        assert out.shape == (0, 2)


class TestDtypeInfo:
    def test_finfo_float32(self) -> None:
        fi = lucid.finfo(lucid.float32)
        assert fi.bits == 32
        assert math.isclose(fi.eps, 1.1920929e-7, rel_tol=1e-3)
        assert fi.dtype == "float32"
        assert fi.smallest_normal == fi.tiny

    def test_finfo_float64(self) -> None:
        fi = lucid.finfo(lucid.float64)
        assert fi.bits == 64

    def test_finfo_rejects_int(self) -> None:
        with pytest.raises(TypeError):
            lucid.finfo(lucid.int32)

    def test_iinfo_int32(self) -> None:
        ii = lucid.iinfo(lucid.int32)
        assert ii.bits == 32
        assert ii.max == 2**31 - 1
        assert ii.min == -(2**31)

    def test_iinfo_rejects_float(self) -> None:
        with pytest.raises(TypeError):
            lucid.iinfo(lucid.float32)


class TestThreadingStubs:
    def test_set_get_intra_op(self) -> None:
        lucid.set_num_threads(8)
        assert lucid.get_num_threads() == 8
        lucid.set_num_threads(2)
        assert lucid.get_num_threads() == 2

    def test_set_get_inter_op(self) -> None:
        lucid.set_num_interop_threads(3)
        assert lucid.get_num_interop_threads() == 3

    def test_invalid_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            lucid.set_num_threads(0)


class TestDeterminism:
    def test_toggle(self) -> None:
        lucid.use_deterministic_algorithms(True)
        assert lucid.are_deterministic_algorithms_enabled() is True
        lucid.use_deterministic_algorithms(False)
        assert lucid.are_deterministic_algorithms_enabled() is False
