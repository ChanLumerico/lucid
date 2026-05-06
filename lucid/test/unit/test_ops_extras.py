"""Tests for the surface-level tensor ops added to fill ``torch`` parity gaps.

Each block exercises one category against either NumPy or the reference
framework so regressions surface immediately.
"""

import math

import numpy as np
import pytest

import lucid


_REF = pytest.importorskip("torch")


def _np(x: object) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


# ── Shape / view ──────────────────────────────────────────────────────────────


class TestShapeView:
    def test_view_alias_for_reshape(self) -> None:
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(_np(lucid.view(t, 4)), _np(t.reshape(4)))

    def test_concat_alias_for_cat(self) -> None:
        a: lucid.Tensor = lucid.tensor([1.0, 2.0])
        b: lucid.Tensor = lucid.tensor([3.0])
        np.testing.assert_allclose(_np(lucid.concat([a, b])), [1.0, 2.0, 3.0])

    def test_narrow_slice(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(_np(lucid.narrow(t, 0, 1, 2)), [2.0, 3.0])

    def test_narrow_out_of_range_raises(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0])
        with pytest.raises(IndexError):
            lucid.narrow(t, 0, 0, 10)

    def test_movedim_swaps_axes(self) -> None:
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(_np(lucid.movedim(t, 0, 1)), [[1.0, 3.0], [2.0, 4.0]])

    def test_unflatten_inverse_of_flatten(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            _np(lucid.unflatten(t, 0, [2, 2])), [[1.0, 2.0], [3.0, 4.0]]
        )


# ── Indexing ─────────────────────────────────────────────────────────────────


class TestIndexing:
    def test_take_flat_indices(self) -> None:
        t: lucid.Tensor = lucid.tensor([[10.0, 20.0], [30.0, 40.0]])
        idx: lucid.Tensor = lucid.tensor([0, 3], dtype=lucid.int64)
        np.testing.assert_allclose(_np(lucid.take(t, idx)), [10.0, 40.0])

    def test_index_select_rows(self) -> None:
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        idx: lucid.Tensor = lucid.tensor([2, 0], dtype=lucid.int64)
        np.testing.assert_allclose(
            _np(lucid.index_select(t, 0, idx)), [[5.0, 6.0], [1.0, 2.0]]
        )

    def test_index_select_int_dtype_required(self) -> None:
        t: lucid.Tensor = lucid.tensor([[1.0]])
        with pytest.raises(TypeError, match="int"):
            lucid.index_select(t, 0, lucid.tensor([0.0]))

    def test_masked_select_returns_flat(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(_np(lucid.masked_select(t, t > 2)), [3.0, 4.0])

    def test_scatter_overwrite(self) -> None:
        t: lucid.Tensor = lucid.zeros(4)
        idx: lucid.Tensor = lucid.tensor([0, 2], dtype=lucid.int64)
        src: lucid.Tensor = lucid.tensor([10.0, 20.0])
        np.testing.assert_allclose(
            _np(lucid.scatter(t, 0, idx, src)), [10.0, 0.0, 20.0, 0.0]
        )

    def test_scatter_add_reduce(self) -> None:
        t: lucid.Tensor = lucid.zeros(4)
        idx: lucid.Tensor = lucid.tensor([0, 0, 1, 2], dtype=lucid.int64)
        src: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            _np(lucid.scatter(t, 0, idx, src, reduce="add")),
            [3.0, 3.0, 4.0, 0.0],
        )

    def test_scatter_unknown_reduce_raises(self) -> None:
        t: lucid.Tensor = lucid.zeros(2)
        with pytest.raises(NotImplementedError):
            lucid.scatter(
                t,
                0,
                lucid.tensor([0], dtype=lucid.int64),
                lucid.tensor([1.0]),
                reduce="multiply",
            )

    def test_kthvalue_matches_sort(self) -> None:
        t: lucid.Tensor = lucid.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
        np.testing.assert_allclose(float(lucid.kthvalue(t, 3).item()), 3.0)


# ── Comparison free functions ────────────────────────────────────────────────


class TestComparison:
    @pytest.mark.parametrize(
        "fn,torch_fn",
        [
            (lucid.eq, _REF.eq),
            (lucid.ne, _REF.ne),
            (lucid.lt, _REF.lt),
            (lucid.le, _REF.le),
            (lucid.gt, _REF.gt),
            (lucid.ge, _REF.ge),
        ],
    )
    def test_pairwise_parity(self, fn: object, torch_fn: object) -> None:
        a_np: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_np: np.ndarray = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        out_l = fn(lucid.tensor(a_np.copy()), lucid.tensor(b_np.copy()))  # type: ignore[operator]
        out_t = torch_fn(_REF.tensor(a_np.copy()), _REF.tensor(b_np.copy()))  # type: ignore[operator]
        np.testing.assert_array_equal(_np(out_l).astype(bool), out_t.numpy().astype(bool))

    def test_isclose_within_tolerance(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0])
        out: lucid.Tensor = lucid.isclose(t, t + 1e-9)
        assert _np(out).astype(bool).all()


# ── Logical / bitwise ────────────────────────────────────────────────────────


class TestLogical:
    def test_logical_and(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            _np(lucid.logical_and(t > 1, t < 4)).astype(bool),
            np.array([False, True, True, False]),
        )

    def test_logical_or(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            _np(lucid.logical_or(t < 2, t > 3)).astype(bool),
            np.array([True, False, False, True]),
        )

    def test_logical_xor(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(
            _np(lucid.logical_xor(t < 2, t < 3)).astype(bool),
            np.array([False, True, False, False]),
        )

    def test_logical_not(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            _np(lucid.logical_not(t > 1)).astype(bool),
            np.array([True, False, False]),
        )

    def test_bitwise_not(self) -> None:
        t: lucid.Tensor = lucid.tensor([0, 1, 2], dtype=lucid.int32)
        np.testing.assert_array_equal(_np(lucid.bitwise_not(t)), [-1, -2, -3])

    @pytest.mark.parametrize(
        "lucid_fn,torch_fn",
        [
            (lucid.bitwise_and, _REF.bitwise_and),
            (lucid.bitwise_or, _REF.bitwise_or),
            (lucid.bitwise_xor, _REF.bitwise_xor),
        ],
    )
    def test_bitwise_pairwise(self, lucid_fn, torch_fn) -> None:  # type: ignore[no-untyped-def]
        a_np: np.ndarray = np.array([5, 3, 12, 7], dtype=np.int32)
        b_np: np.ndarray = np.array([3, 5, 10, 6], dtype=np.int32)
        out_l: lucid.Tensor = lucid_fn(  # type: ignore[operator]
            lucid.tensor(a_np.copy(), dtype=lucid.int32),
            lucid.tensor(b_np.copy(), dtype=lucid.int32),
        )
        out_t = torch_fn(_REF.tensor(a_np.copy()), _REF.tensor(b_np.copy()))  # type: ignore[operator]
        np.testing.assert_array_equal(_np(out_l), out_t.numpy())


# ── Math (unary) ─────────────────────────────────────────────────────────────


class TestUnaryMath:
    @pytest.mark.parametrize(
        "lucid_fn,torch_fn,scale",
        [
            (lucid.asin, _REF.asin, 0.1),
            (lucid.acos, _REF.acos, 0.1),
            (lucid.atan, _REF.atan, 1.0),
            (lucid.log10, _REF.log10, 1.0),
            (lucid.log1p, _REF.log1p, 1.0),
            (lucid.exp2, _REF.exp2, 1.0),
            (lucid.trunc, _REF.trunc, 1.7),
            (lucid.frac, _REF.frac, 1.7),
        ],
    )
    def test_parity(self, lucid_fn, torch_fn, scale: float) -> None:  # type: ignore[no-untyped-def]
        v: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * scale
        out_l: lucid.Tensor = lucid_fn(lucid.tensor(v.copy()))  # type: ignore[operator]
        out_t = torch_fn(_REF.tensor(v.copy()))  # type: ignore[operator]
        np.testing.assert_allclose(_np(out_l), out_t.numpy(), atol=1e-5)


# ── Math (binary) ────────────────────────────────────────────────────────────


class TestBinaryMath:
    def test_atan2_quadrant_corrections(self) -> None:
        y: lucid.Tensor = lucid.tensor([1.0, -1.0, -1.0, 1.0])
        x: lucid.Tensor = lucid.tensor([1.0, 1.0, -1.0, -1.0])
        out: np.ndarray = _np(lucid.atan2(y, x))
        np.testing.assert_allclose(
            out,
            _REF.atan2(_REF.tensor([1.0, -1.0, -1.0, 1.0]), _REF.tensor([1.0, 1.0, -1.0, -1.0])).numpy(),
            atol=1e-5,
        )

    def test_fmod_keeps_dividend_sign(self) -> None:
        x: lucid.Tensor = lucid.tensor([-3.5, 3.5])
        y: lucid.Tensor = lucid.tensor([2.0, 2.0])
        np.testing.assert_allclose(
            _np(lucid.fmod(x, y)), _REF.fmod(_REF.tensor([-3.5, 3.5]), _REF.tensor([2.0, 2.0])).numpy()
        )

    def test_remainder_keeps_divisor_sign(self) -> None:
        x: lucid.Tensor = lucid.tensor([-3.5, 3.5])
        y: lucid.Tensor = lucid.tensor([2.0, 2.0])
        np.testing.assert_allclose(
            _np(lucid.remainder(x, y)),
            _REF.remainder(_REF.tensor([-3.5, 3.5]), _REF.tensor([2.0, 2.0])).numpy(),
        )

    def test_hypot_pythagoras(self) -> None:
        np.testing.assert_allclose(
            float(lucid.hypot(lucid.tensor([3.0]), lucid.tensor([4.0])).item()), 5.0
        )

    def test_logaddexp_stable(self) -> None:
        a: lucid.Tensor = lucid.tensor([1.0, 1000.0])
        b: lucid.Tensor = lucid.tensor([2.0, 1000.0])
        out: np.ndarray = _np(lucid.logaddexp(a, b))
        # No overflow: 1000-pair must still be finite.
        assert math.isfinite(float(out[1]))


# ── Reductions ───────────────────────────────────────────────────────────────


class TestLogSumExp:
    def test_against_torch(self) -> None:
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        out: float = float(lucid.logsumexp(t).item())
        ref: float = float(_REF.logsumexp(_REF.tensor([1.0, 2.0, 3.0, 4.0]), 0).item())
        assert abs(out - ref) < 1e-5

    def test_keepdim(self) -> None:
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        out: lucid.Tensor = lucid.logsumexp(t, dim=0, keepdim=True)
        assert tuple(out.shape) == (1, 2)


# ── Linear algebra ───────────────────────────────────────────────────────────


class TestLinAlg:
    def test_mm_2d_only(self) -> None:
        a: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(_np(lucid.mm(a, a)), [[7.0, 10.0], [15.0, 22.0]])

    def test_mm_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            lucid.mm(lucid.tensor([1.0, 2.0]), lucid.tensor([3.0, 4.0]))

    def test_bmm_batched(self) -> None:
        a: lucid.Tensor = lucid.tensor([[[1.0, 2.0]]])
        b: lucid.Tensor = lucid.tensor([[[3.0], [4.0]]])
        np.testing.assert_allclose(_np(lucid.bmm(a, b)), [[[11.0]]])

    def test_einsum_top_level(self) -> None:
        a: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        out: float = float(lucid.einsum("ij,ij->", a, a).item())
        assert out == 30.0  # 1+4+9+16

    def test_einsum_via_einops_package(self) -> None:
        # The user explicitly wanted ``lucid.einops.einsum`` to remain a
        # valid entry point even after the top-level ``lucid.einsum`` lands.
        a: lucid.Tensor = lucid.tensor([1.0, 2.0])
        b: lucid.Tensor = lucid.tensor([3.0, 4.0])
        assert float(lucid.einops.einsum("i,i->", a, b).item()) == 11.0

    def test_top_level_norm(self) -> None:
        # ``lucid.norm`` is a top-level alias of ``lucid.linalg.norm``.
        assert float(lucid.norm(lucid.tensor([3.0, 4.0])).item()) == 5.0

    def test_top_level_cross(self) -> None:
        a: lucid.Tensor = lucid.tensor([1.0, 0.0, 0.0])
        b: lucid.Tensor = lucid.tensor([0.0, 1.0, 0.0])
        np.testing.assert_allclose(_np(lucid.cross(a, b)), [0.0, 0.0, 1.0])

    def test_kron_product(self) -> None:
        a: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(_np(lucid.kron(a, a)), _REF.kron(_REF.tensor([[1.0, 2.0], [3.0, 4.0]]), _REF.tensor([[1.0, 2.0], [3.0, 4.0]])).numpy())


# ── Stats / search ───────────────────────────────────────────────────────────


class TestStats:
    def test_searchsorted(self) -> None:
        seq: lucid.Tensor = lucid.tensor([1.0, 3.0, 5.0, 7.0])
        out: np.ndarray = _np(lucid.searchsorted(seq, lucid.tensor([0.0, 2.0, 4.0, 8.0])))
        np.testing.assert_array_equal(out, [0, 1, 2, 4])

    def test_bucketize_alias(self) -> None:
        boundaries: lucid.Tensor = lucid.tensor([1.0, 3.0, 5.0])
        np.testing.assert_array_equal(
            _np(lucid.bucketize(lucid.tensor([0.5, 2.0, 4.0, 6.0]), boundaries)),
            [0, 1, 2, 3],
        )

    def test_histc(self) -> None:
        # Match what the reference framework returns for the same call —
        # 4 equal bins on [0, 5] place each input in a distinct bin.
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        out: np.ndarray = _np(lucid.histc(t, bins=4, min=0.0, max=5.0))
        ref: np.ndarray = _REF.histc(
            _REF.tensor([1.0, 2.0, 3.0, 4.0]), bins=4, min=0.0, max=5.0
        ).numpy()
        np.testing.assert_allclose(out, ref)

    def test_cartesian_prod(self) -> None:
        a: lucid.Tensor = lucid.tensor([1.0, 2.0])
        b: lucid.Tensor = lucid.tensor([3.0, 4.0])
        np.testing.assert_array_equal(
            _np(lucid.cartesian_prod(a, b)),
            [[1.0, 3.0], [1.0, 4.0], [2.0, 3.0], [2.0, 4.0]],
        )
