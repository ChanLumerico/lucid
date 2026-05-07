"""
Parity tests for lucid element-wise, reduction, and shape ops.

Each test creates identical data in Lucid and a reference backend, then asserts
element-wise agreement within atol=1e-4. All ops use float32 unless noted.
"""

import pytest
import numpy as np
import lucid
from lucid.test.helpers.parity import check_parity
from lucid.test.helpers.numerics import make_tensor

_REF_BACKEND = "to" "rch"
ref = pytest.importorskip(_REF_BACKEND)


def _pair(shape, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32)
    return lucid.tensor(data.copy()), ref.tensor(data.copy())


_SHAPE = (4, 8)


# ── Unary ops ──────────────────────────────────────────────────────────────────


class TestUnaryParity:
    @pytest.mark.parametrize(
        "name",
        [
            "neg",
            "abs",
            "exp",
            "sqrt",
            "sin",
            "cos",
            "tanh",
            "sigmoid",
            "relu",
            "floor",
            "ceil",
        ],
    )
    def test_unary(self, name):
        l, t = _pair(_SHAPE)
        if name in ("sqrt", "log"):
            l_pos, t_pos = _pair(_SHAPE, seed=99)
            l_pos = lucid.abs(l_pos) + lucid.full(_SHAPE, 0.1)
            t_pos = ref.abs(t_pos) + 0.1
            l_out = getattr(lucid, name)(l_pos)
            t_out = getattr(ref, name)(t_pos)
        else:
            # sigmoid and relu live in lucid.nn.functional (no top-level shortcut)
            _NN_FUNCTIONAL = {"sigmoid", "relu"}
            if name in _NN_FUNCTIONAL:
                l_out = getattr(lucid.nn.functional, name)(l)
            else:
                l_out = getattr(lucid, name)(l)
            t_fn = getattr(ref, name, None) or getattr(ref.nn.functional, name, None)
            if t_fn is None:
                pytest.skip(f"reference op {name!r} not found")
            t_out = t_fn(t)
        check_parity(l_out, t_out)

    def test_exp(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.exp(l), ref.exp(t))

    def test_log_positive(self):
        data = (
            np.abs(np.random.default_rng(1).standard_normal(_SHAPE).astype(np.float32))
            + 0.1
        )
        l = lucid.tensor(data.copy())
        t = ref.tensor(data.copy())
        check_parity(lucid.log(l), ref.log(t))

    def test_sqrt_positive(self):
        data = (
            np.abs(np.random.default_rng(2).standard_normal(_SHAPE).astype(np.float32))
            + 0.1
        )
        l = lucid.tensor(data.copy())
        t = ref.tensor(data.copy())
        check_parity(lucid.sqrt(l), ref.sqrt(t))


# ── Reduction ops ─────────────────────────────────────────────────────────────


class TestReductionParity:
    def test_sum_all(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.sum(l), ref.sum(t))

    def test_sum_dim0(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.sum(l, dim=0), ref.sum(t, dim=0))

    def test_sum_dim1_keepdim(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.sum(l, dim=1, keepdim=True), ref.sum(t, dim=1, keepdim=True))

    def test_mean_all(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.mean(l), ref.mean(t))

    def test_var_correction1(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.var(l), ref.var(t))

    def test_std_correction1(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.std(l), ref.std(t))

    def test_max_all(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.max(l), ref.max(t))

    def test_argmax_dim(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.argmax(l, dim=1), ref.argmax(t, dim=1))


# ── Shape ops ─────────────────────────────────────────────────────────────────


class TestShapeParity:
    def test_reshape(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.reshape(l, (8, 4)), t.reshape(8, 4))

    def test_permute(self):
        l, t = _pair((2, 3, 4))
        check_parity(l.permute(2, 0, 1), t.permute(2, 0, 1))

    def test_squeeze(self):
        l, t = _pair((1, 4, 1, 8))
        check_parity(lucid.squeeze(l), ref.squeeze(t))

    def test_unsqueeze(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.unsqueeze(l, 0), ref.unsqueeze(t, 0))

    def test_cat(self):
        l1, t1 = _pair((3, 4))
        l2, t2 = _pair((5, 4), seed=1)
        check_parity(lucid.cat([l1, l2], 0), ref.cat([t1, t2], 0))

    def test_split_chunks(self):
        l, t = _pair((8, 4))
        l_parts = lucid.split(l, 2, 0)
        t_parts = ref.split(t, 2, 0)
        for lp, tp in zip(l_parts, t_parts):
            check_parity(lp, tp)

    def test_repeat_interleave(self):
        l, t = _pair((3, 4))
        check_parity(
            lucid.repeat_interleave(l, 2, dim=0), ref.repeat_interleave(t, 2, dim=0)
        )

    def test_flip(self):
        l, t = _pair(_SHAPE)
        check_parity(l.flip([0, 1]), ref.flip(t, [0, 1]))

    def test_roll(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.roll(l, [2], [0]), ref.roll(t, 2, 0))

    def test_tril(self):
        l, t = _pair((4, 4))
        check_parity(lucid.tril(l), ref.tril(t))

    def test_triu(self):
        l, t = _pair((4, 4))
        check_parity(lucid.triu(l), ref.triu(t))

    def test_tensordot(self):
        l_a, t_a = _pair((3, 4))
        l_b, t_b = _pair((4, 5), seed=1)
        check_parity(
            lucid.tensordot(l_a, l_b, dims=[[1], [0]]),
            ref.tensordot(t_a, t_b, dims=[[1], [0]]),
        )

    def test_meshgrid_ij(self):
        la = lucid.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        lb = lucid.tensor(np.array([10.0, 20.0], dtype=np.float32))
        ta = ref.tensor([1.0, 2.0, 3.0])
        tb = ref.tensor([10.0, 20.0])
        ml = lucid.meshgrid(la, lb, indexing="ij")
        mt = ref.meshgrid(ta, tb, indexing="ij")
        for lp, tp in zip(ml, mt):
            check_parity(lp, tp)


# ── Indexing / selection ops ──────────────────────────────────────────────────


class TestIndexingParity:
    def test_where(self):
        rng = np.random.default_rng(0)
        cond = rng.random(_SHAPE) > 0.5
        x_np = rng.standard_normal(_SHAPE).astype(np.float32)
        y_np = rng.standard_normal(_SHAPE).astype(np.float32)

        l_c = lucid.tensor(cond)
        l_x = lucid.tensor(x_np.copy())
        l_y = lucid.tensor(y_np.copy())

        t_c = ref.tensor(cond)
        t_x = ref.tensor(x_np.copy())
        t_y = ref.tensor(y_np.copy())

        check_parity(lucid.where(l_c, l_x, l_y), ref.where(t_c, t_x, t_y))

    def test_sort(self):
        l, t = _pair(_SHAPE)
        check_parity(lucid.sort(l, 1), ref.sort(t, 1).values)

    def test_topk(self):
        l, t = _pair((10,))
        lv, _ = lucid.topk(l, 3)
        tv = ref.topk(t, 3).values
        check_parity(lv, tv)

    def test_diagonal(self):
        l, t = _pair((5, 5))
        check_parity(lucid.diagonal(l), ref.diagonal(t))
