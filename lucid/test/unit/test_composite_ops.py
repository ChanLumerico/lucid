"""Composite ops layered on engine primitives — see ``lucid/_ops/composite/``.

The composite package groups Python-level ops by category
(``elementwise``, ``reductions``, ``shape``, ``blas``, ``predicates``,
``dtype``, ``constants``).  Each test class below mirrors one of those
files.

Coverage focuses on:
* every name re-exported via ``COMPOSITE_NAMES`` is reachable through
  ``lucid.X`` (the lazy-loader plumbing works);
* numerical correctness on a small representative input;
* edge cases that the composite explicitly handles (NaN propagation,
  ``sinc`` at zero, ``isin`` with non-overlapping inputs, etc.).
"""

import math

import pytest

import lucid
from lucid._ops.composite import COMPOSITE_NAMES

# ── Plumbing ───────────────────────────────────────────────────────────────


class TestCompositeSurface:
    def test_every_name_resolves_at_top_level(self) -> None:
        missing = [n for n in COMPOSITE_NAMES if not hasattr(lucid, n)]
        assert missing == [], f"composite names not exposed at top level: {missing}"


# ── constants.py ───────────────────────────────────────────────────────────


class TestConstants:
    def test_math_constants(self) -> None:
        assert lucid.pi == math.pi
        assert lucid.e == math.e
        assert lucid.inf == math.inf
        assert math.isnan(lucid.nan)

    def test_newaxis_is_none(self) -> None:
        # ``x[:, lucid.newaxis]`` — same idiom as NumPy / PyTorch.
        assert lucid.newaxis is None


# ── elementwise.py ─────────────────────────────────────────────────────────


class TestElementwiseAliases:
    def test_absolute_negative_positive(self) -> None:
        x = lucid.tensor([-2.0, 3.0, -5.0])
        assert lucid.absolute(x).sum().item() == 10.0
        assert lucid.negative(x).sum().item() == 4.0
        assert (lucid.positive(x) - x).abs().max().item() == 0.0

    def test_subtract_with_alpha(self) -> None:
        a = lucid.tensor([10.0])
        b = lucid.tensor([2.0])
        assert lucid.subtract(a, b).item() == 8.0
        assert lucid.subtract(a, b, alpha=2.0).item() == 6.0

    def test_multiply_divide_aliases(self) -> None:
        a = lucid.tensor([6.0])
        b = lucid.tensor([2.0])
        assert lucid.multiply(a, b).item() == 12.0
        assert lucid.divide(a, b).item() == 3.0
        assert lucid.true_divide(a, b).item() == 3.0

    def test_rsub(self) -> None:
        a = lucid.tensor([3.0])
        b = lucid.tensor([10.0])
        assert lucid.rsub(a, b).item() == 7.0
        assert lucid.rsub(a, b, alpha=2.0).item() == 4.0


class TestElementwiseInverseHyperbolic:
    def test_acosh_asinh_atanh(self) -> None:
        assert (
            pytest.approx(math.acosh(2.0), abs=1e-4)
            == lucid.acosh(lucid.tensor([2.0])).item()
        )
        assert (
            pytest.approx(math.asinh(1.0), abs=1e-4)
            == lucid.asinh(lucid.tensor([1.0])).item()
        )
        assert (
            pytest.approx(math.atanh(0.5), abs=1e-4)
            == lucid.atanh(lucid.tensor([0.5])).item()
        )

    def test_arctan2(self) -> None:
        out = lucid.arctan2(lucid.tensor([1.0]), lucid.tensor([1.0])).item()
        assert pytest.approx(math.pi / 4, abs=1e-4) == out


class TestElementwiseSpecials:
    def test_expm1(self) -> None:
        assert lucid.expm1(lucid.tensor([0.0])).item() == 0.0
        assert (
            pytest.approx(math.expm1(1.0), abs=1e-3)
            == lucid.expm1(lucid.tensor([1.0])).item()
        )

    def test_sinc_zero_branch(self) -> None:
        # The ``where`` guard must hide the 0/0 division at x = 0.
        assert lucid.sinc(lucid.tensor([0.0])).item() == 1.0
        assert (
            pytest.approx(2.0 / math.pi, abs=1e-4)
            == lucid.sinc(lucid.tensor([0.5])).item()
        )

    def test_heaviside(self) -> None:
        out = lucid.heaviside(lucid.tensor([-1.0, 0.0, 2.0]), 0.5)
        assert (out - lucid.tensor([0.0, 0.5, 1.0])).abs().max().item() < 1e-6

    def test_xlogy_zero_zero_is_zero(self) -> None:
        # PyTorch convention 0 * log(0) = 0 — must not be NaN.
        assert lucid.xlogy(lucid.tensor([0.0]), lucid.tensor([0.0])).item() == 0.0

    def test_logit(self) -> None:
        # logit(0.5) = 0
        assert pytest.approx(0.0, abs=1e-6) == lucid.logit(lucid.tensor([0.5])).item()

    def test_signbit(self) -> None:
        out = lucid.signbit(lucid.tensor([-1.0, 0.0, 1.0]))
        as_int = out.to(dtype=lucid.float32).sum().item()
        assert as_int == 1  # only -1.0 has sign bit set

    def test_fmax_fmin_nan_propagation(self) -> None:
        a = lucid.tensor([1.0, float("nan"), 3.0])
        b = lucid.tensor([2.0, 5.0, float("nan")])
        # Where one side is NaN, the non-NaN value should win.
        fmx = lucid.fmax(a, b)
        fmn = lucid.fmin(a, b)
        assert (fmx - lucid.tensor([2.0, 5.0, 3.0])).abs().max().item() < 1e-6
        assert (fmn - lucid.tensor([1.0, 5.0, 3.0])).abs().max().item() < 1e-6

    def test_float_power_promotes_to_f64(self) -> None:
        out = lucid.float_power(lucid.tensor([2.0], dtype=lucid.float32), 3.0)
        assert out.dtype == lucid.float64


# ── reductions.py ──────────────────────────────────────────────────────────


class TestNanReductions:
    def test_nansum_ignores_nan(self) -> None:
        x = lucid.tensor([1.0, float("nan"), 3.0])
        assert lucid.nansum(x).item() == 4.0

    def test_nanmean_uses_non_nan_count(self) -> None:
        x = lucid.tensor([1.0, float("nan"), 3.0])
        assert lucid.nanmean(x).item() == 2.0

    def test_nanmedian_odd_count(self) -> None:
        x = lucid.tensor([float("nan"), 1.0, 3.0, 5.0, float("nan")])
        # Lower median of [1, 3, 5] is 3.
        assert lucid.nanmedian(x).item() == 3.0


# ── shape.py ───────────────────────────────────────────────────────────────


class TestShapeSwaps:
    def test_swapaxes_swapdims(self) -> None:
        x = lucid.randn(2, 3, 4)
        assert lucid.swapaxes(x, 0, 2).shape == (4, 3, 2)
        assert lucid.swapdims(x, 1, 2).shape == (2, 4, 3)

    def test_moveaxis(self) -> None:
        x = lucid.randn(2, 3, 4)
        assert lucid.moveaxis(x, 0, 2).shape == (3, 4, 2)

    def test_adjoint_swaps_last_two(self) -> None:
        x = lucid.randn(2, 3, 4)
        assert lucid.adjoint(x).shape == (2, 4, 3)
        with pytest.raises(ValueError, match="2 dimensions"):
            lucid.adjoint(lucid.tensor([1.0]))

    def test_t_2d_only(self) -> None:
        m = lucid.randn(3, 4)
        assert lucid.t(m).shape == (4, 3)
        # 1-D passes through.
        v = lucid.tensor([1.0, 2.0, 3.0])
        assert (lucid.t(v) - v).abs().max().item() == 0.0
        with pytest.raises(RuntimeError, match="<= 2"):
            lucid.t(lucid.randn(2, 3, 4))


class TestShapeStacks:
    def test_column_stack_promotes_1d(self) -> None:
        out = lucid.column_stack(
            [lucid.tensor([1.0, 2.0, 3.0]), lucid.tensor([4.0, 5.0, 6.0])]
        )
        assert out.shape == (3, 2)

    def test_row_stack_alias_of_vstack(self) -> None:
        a = lucid.tensor([[1.0, 2.0]])
        b = lucid.tensor([[3.0, 4.0]])
        assert (
            lucid.row_stack([a, b]) - lucid.vstack([a, b])
        ).abs().max().item() == 0.0

    def test_dstack_promotes_lower_rank(self) -> None:
        out = lucid.dstack([lucid.zeros(3, 4), lucid.ones(3, 4)])
        assert out.shape == (3, 4, 2)

    def test_atleast_family(self) -> None:
        scalar = lucid.tensor(5.0)
        assert lucid.atleast_1d(scalar).shape == (1,)
        assert lucid.atleast_2d(scalar).shape == (1, 1)
        assert lucid.atleast_3d(scalar).shape == (1, 1, 1)
        # Multi-input returns a tuple.
        out = lucid.atleast_2d(scalar, lucid.tensor([1.0, 2.0]))
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == (1, 1) and out[1].shape == (1, 2)


class TestShapeSplits:
    def test_tensor_split_sections(self) -> None:
        parts = lucid.tensor_split(lucid.arange(7).reshape(7), 3)
        # 7 elements into 3 sections → ceil(7/3)=3, then 2, then 2.
        assert [p.shape[0] for p in parts] == [3, 2, 2]

    def test_tensor_split_indices(self) -> None:
        parts = lucid.tensor_split(lucid.arange(6), [2, 4])
        assert [int(p.numel()) for p in parts] == [2, 2, 2]

    def test_vsplit_hsplit_dsplit(self) -> None:
        v = lucid.vsplit(lucid.zeros(6, 4), 2)
        assert v[0].shape == (3, 4)
        h = lucid.hsplit(lucid.zeros(4, 6), 2)
        assert h[0].shape == (4, 3)
        d = lucid.dsplit(lucid.zeros(2, 3, 6), 2)
        assert d[0].shape == (2, 3, 3)


class TestShapeMisc:
    def test_take_along_dim(self) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = lucid.tensor([[0, 2], [1, 0]], dtype=lucid.int64)
        out = lucid.take_along_dim(x, idx, 1)
        assert out.shape == (2, 2)
        assert (out - lucid.tensor([[1.0, 3.0], [5.0, 4.0]])).abs().max().item() == 0.0

    def test_vander(self) -> None:
        out = lucid.vander(lucid.tensor([1.0, 2.0, 3.0]), N=4)
        # Last column (decreasing power) should be x^0 = 1.
        assert (out[:, 3] - lucid.tensor([1.0, 1.0, 1.0])).abs().max().item() < 1e-6
        # Increasing variant.
        out_inc = lucid.vander(lucid.tensor([1.0, 2.0, 3.0]), N=4, increasing=True)
        assert (out_inc[:, 0] - lucid.tensor([1.0, 1.0, 1.0])).abs().max().item() < 1e-6

    def test_rot90_k_cycles(self) -> None:
        x = lucid.arange(6, dtype=lucid.float32).reshape(2, 3)
        # k=4 should be identity.
        assert (lucid.rot90(x, 4) - x).abs().max().item() == 0.0
        # k=2 == flip on both dims.
        assert (lucid.rot90(x, 2) - lucid.flip(x, [0, 1])).abs().max().item() == 0.0


# ── blas.py ────────────────────────────────────────────────────────────────


class TestBlasComposites:
    def test_addmm_with_alpha_beta(self) -> None:
        c = lucid.ones(2, 4)
        a = lucid.zeros(2, 3)  # mat1 zero → matmul=0
        b = lucid.randn(3, 4)
        out = lucid.addmm(c, a, b, beta=2.0, alpha=3.0)
        # 2*1 + 3*0 = 2 everywhere
        assert (out - lucid.full_like(out, 2.0)).abs().max().item() < 1e-6

    def test_addbmm_collapses_batch(self) -> None:
        out = lucid.addbmm(
            lucid.zeros(2, 4), lucid.randn(5, 2, 3), lucid.randn(5, 3, 4)
        )
        assert out.shape == (2, 4)

    def test_baddbmm_keeps_batch(self) -> None:
        out = lucid.baddbmm(
            lucid.zeros(5, 2, 4), lucid.randn(5, 2, 3), lucid.randn(5, 3, 4)
        )
        assert out.shape == (5, 2, 4)

    def test_addmv_addr(self) -> None:
        assert lucid.addmv(lucid.zeros(2), lucid.randn(2, 3), lucid.randn(3)).shape == (
            2,
        )
        assert lucid.addr(lucid.zeros(3, 4), lucid.randn(3), lucid.randn(4)).shape == (
            3,
            4,
        )

    def test_addcmul_addcdiv(self) -> None:
        out = lucid.addcmul(
            lucid.zeros(3),
            lucid.tensor([1.0, 2.0, 3.0]),
            lucid.tensor([2.0, 2.0, 2.0]),
        )
        assert out.sum().item() == 12.0
        out = lucid.addcdiv(
            lucid.zeros(3),
            lucid.tensor([10.0, 20.0, 30.0]),
            lucid.tensor([2.0, 2.0, 2.0]),
        )
        assert out.sum().item() == 30.0

    def test_mv_ger(self) -> None:
        assert lucid.mv(lucid.randn(2, 3), lucid.randn(3)).shape == (2,)
        assert lucid.linalg.outer(lucid.randn(3), lucid.randn(4)).shape == (3, 4)

    def test_block_diag(self) -> None:
        out = lucid.block_diag(lucid.eye(2), lucid.ones(3, 3))
        assert out.shape == (5, 5)
        # Off-diagonal must be zero.
        assert out[0:2, 2:5].abs().max().item() == 0.0
        assert out[2:5, 0:2].abs().max().item() == 0.0


# ── predicates.py ──────────────────────────────────────────────────────────


class TestPredicates:
    def test_numel(self) -> None:
        assert lucid.numel(lucid.zeros(3, 4)) == 12

    def test_is_storage_always_false(self) -> None:
        assert lucid.is_storage(lucid.tensor([1.0])) is False

    def test_is_nonzero_scalar_only(self) -> None:
        assert lucid.is_nonzero(lucid.tensor([5.0])) is True
        assert lucid.is_nonzero(lucid.tensor([0.0])) is False
        with pytest.raises(RuntimeError, match="numel == 1"):
            lucid.is_nonzero(lucid.tensor([1.0, 2.0]))

    def test_is_same_size(self) -> None:
        assert lucid.is_same_size(lucid.zeros(3, 4), lucid.zeros(3, 4)) is True
        assert lucid.is_same_size(lucid.zeros(3, 4), lucid.zeros(3, 5)) is False

    def test_is_neg_is_conj_stubs(self) -> None:
        x = lucid.tensor([1.0])
        assert lucid.is_neg(x) is False
        assert lucid.is_conj(x) is False

    def test_isin(self) -> None:
        out = lucid.isin(lucid.tensor([1.0, 2.0, 3.0, 4.0]), lucid.tensor([2.0, 4.0]))
        assert out.to(dtype=lucid.float32).sum().item() == 2

    def test_isin_invert(self) -> None:
        out = lucid.isin(
            lucid.tensor([1.0, 2.0, 3.0]), lucid.tensor([2.0]), invert=True
        )
        assert out.to(dtype=lucid.float32).sum().item() == 2

    def test_isneginf_isposinf_isreal(self) -> None:
        x = lucid.tensor([float("-inf"), 1.0, float("inf")])
        assert lucid.isneginf(x).to(dtype=lucid.float32).sum().item() == 1
        assert lucid.isposinf(x).to(dtype=lucid.float32).sum().item() == 1
        # All real → all True.
        assert lucid.isreal(x).to(dtype=lucid.float32).sum().item() == 3

    def test_conj_is_identity_for_real(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        assert (lucid.conj(x) - x).abs().max().item() == 0.0
        assert (lucid.conj_physical(x) - x).abs().max().item() == 0.0
        assert lucid.resolve_conj(x) is x
        assert lucid.resolve_neg(x) is x


# ── dtype.py ───────────────────────────────────────────────────────────────


class TestDtypePromotion:
    def test_result_type_widens(self) -> None:
        a = lucid.zeros(1, dtype=lucid.float32)
        b = lucid.zeros(1, dtype=lucid.float64)
        assert lucid.result_type(a, b) == lucid.float64

    def test_result_type_kind_priority(self) -> None:
        # int + float → float wins regardless of width.
        a = lucid.zeros(1, dtype=lucid.int64)
        b = lucid.zeros(1, dtype=lucid.float32)
        assert lucid.result_type(a, b) == lucid.float32

    def test_promote_types_idempotent_on_equal(self) -> None:
        assert lucid.promote_types(lucid.float32, lucid.float32) == lucid.float32

    def test_can_cast_widening_only(self) -> None:
        assert lucid.can_cast(lucid.float32, lucid.float64) is True
        assert lucid.can_cast(lucid.float64, lucid.float32) is False

    def test_result_type_with_python_scalar(self) -> None:
        a = lucid.zeros(1, dtype=lucid.float32)
        # A non-tensor argument falls through to ``a``'s dtype.
        assert lucid.result_type(a, 1.0) == lucid.float32
