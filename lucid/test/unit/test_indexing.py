"""Unit tests for indexing ops: gather, scatter_add, where, sort, topk, nonzero, diagonal."""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor, make_int_tensor


class TestGather:
    def test_gather_1d(self):
        t = lucid.tensor([10.0, 20.0, 30.0, 40.0])
        idx = make_int_tensor((3,), low=0, high=4, seed=1)
        out = lucid.gather(t, idx, 0)
        assert out.shape == (3,)

    def test_gather_2d_axis1(self):
        t = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # gather col indices
        idx = make_int_tensor((2, 2), low=0, high=3, seed=0)
        out = lucid.gather(t, idx, 1)
        assert out.shape == (2, 2)


class TestScatterAdd:
    def test_scatter_add_basic(self):
        base = lucid.zeros(4)
        idx = lucid.tensor([0, 2, 0, 1], dtype=lucid.int32)
        src = lucid.tensor([1.0, 1.0, 2.0, 3.0])
        out = lucid.scatter_add(base, 0, idx, src)
        expected = lucid.tensor([3.0, 3.0, 1.0, 0.0])
        assert_close(out, expected)


class TestWhere:
    def test_where_basic(self):
        cond = lucid.tensor([True, False, True])
        x = lucid.tensor([1.0, 2.0, 3.0])
        y = lucid.tensor([10.0, 20.0, 30.0])
        out = lucid.where(cond, x, y)
        assert_close(out, lucid.tensor([1.0, 20.0, 3.0]))

    def test_where_shape_preserved(self):
        cond = make_tensor((3, 4)) > 0
        x = make_tensor((3, 4), seed=1)
        y = make_tensor((3, 4), seed=2)
        out = lucid.where(cond, x, y)
        assert out.shape == (3, 4)


class TestSort:
    def test_sort_ascending(self):
        t = lucid.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
        out = lucid.sort(t, 0)
        arr = out.numpy()
        assert (np.diff(arr) >= 0).all()

    def test_argsort(self):
        t = lucid.tensor([3.0, 1.0, 2.0])
        idx = lucid.argsort(t, 0)
        assert int(idx[0].item()) == 1  # smallest is index 1

    def test_topk(self):
        t = lucid.tensor([3.0, 1.0, 5.0, 2.0, 4.0])
        vals, idxs = lucid.topk(t, 3)
        arr = vals.numpy()
        assert len(arr) == 3
        assert arr[0] == 5.0  # largest first


class TestNonzero:
    def test_nonzero_1d(self):
        t = lucid.tensor([0.0, 1.0, 0.0, 2.0])
        idx = lucid.nonzero(t)
        assert idx.shape[0] == 2


class TestDiagonal:
    def test_diagonal_square(self):
        t = lucid.eye(3)
        d = lucid.diagonal(t)
        assert_close(d, lucid.ones(3))

    def test_diagonal_shape(self):
        t = make_tensor((4, 5))
        d = lucid.diagonal(t)
        assert d.shape == (4,)


class TestMaskedFill:
    def test_masked_fill_zeros(self):
        t = lucid.ones(4)
        mask = lucid.tensor([True, False, True, False])
        out = lucid.masked_fill(t, mask, 0.0)
        expected = lucid.tensor([0.0, 1.0, 0.0, 1.0])
        assert_close(out, expected)


class TestRoll:
    def test_roll_right(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        out = lucid.roll(t, [1], [0])
        expected = lucid.tensor([4.0, 1.0, 2.0, 3.0])
        assert_close(out, expected)

    def test_roll_left(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        out = lucid.roll(t, [-1], [0])
        expected = lucid.tensor([2.0, 3.0, 4.0, 1.0])
        assert_close(out, expected)


class TestFlip:
    def test_flip_1d(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        out = t.flip([0])
        assert_close(out, lucid.tensor([3.0, 2.0, 1.0]))

    def test_flip_2d(self):
        t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = t.flip([0])
        expected = lucid.tensor([[3.0, 4.0], [1.0, 2.0]])
        assert_close(out, expected)

    def test_flip_backward_1d(self):
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out: lucid.Tensor = t.flip([0])
        out.sum().backward()
        assert t.grad is not None
        np.testing.assert_allclose(t.grad.numpy(), np.ones(3))

    def test_flip_backward_preserves_per_element_grad(self):
        # grad of out wrt input is a flipped permutation — feeding a non-uniform
        # upstream grad should arrive flipped at the input.
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        out: lucid.Tensor = t.flip([0])
        upstream: lucid.Tensor = lucid.tensor([10.0, 20.0, 30.0, 40.0])
        (out * upstream).sum().backward()
        np.testing.assert_allclose(t.grad.numpy(), np.array([40.0, 30.0, 20.0, 10.0]))

    def test_flip_backward_multi_axis(self):
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        out: lucid.Tensor = t.flip([0, 1])
        out.sum().backward()
        np.testing.assert_allclose(t.grad.numpy(), np.ones((2, 2)))

    def test_flip_top_level_function(self):
        # Top-level ``lucid.flip`` matches the Tensor method.
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            lucid.flip(t, [0]).numpy(), t.flip([0]).numpy()
        )

    def test_flip_accepts_int_dim(self):
        # ``flip(t, 0)`` and ``flip(t, [0])`` must agree.
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(
            lucid.flip(t, 0).numpy(), lucid.flip(t, [0]).numpy()
        )

    def test_fliplr(self):
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(
            lucid.fliplr(t).numpy(),
            np.array([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]),
        )

    def test_fliplr_rejects_1d(self):
        t: lucid.Tensor = lucid.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="2-D"):
            lucid.fliplr(t)

    def test_flipud(self):
        t: lucid.Tensor = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(
            lucid.flipud(t).numpy(),
            np.array([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]]),
        )


class TestPad:
    def test_pad_1d(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        out = lucid.pad(t, (1, 1))
        assert out.shape == (5,)
        assert float(out[0].item()) == 0.0
        assert float(out[4].item()) == 0.0

    def test_pad_2d(self):
        t = make_tensor((3, 4))
        out = lucid.pad(t, (1, 1, 2, 2))
        assert out.shape == (7, 6)
