"""Unit tests for shape manipulation ops."""

import pytest
import numpy as np
import lucid
from lucid.test._comparison import assert_close
from lucid.test.helpers.numerics import make_tensor


class TestReshape:
    def test_basic(self):
        t = make_tensor((2, 6))
        r = lucid.reshape(t, (3, 4))
        assert r.shape == (3, 4)

    def test_wildcard(self):
        t = make_tensor((2, 3, 4))
        r = t.reshape(-1)
        assert r.shape == (24,)

    def test_values_unchanged(self):
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        r = t.reshape(2, 2)
        assert_close(r.reshape(-1), t)

    def test_view_method(self):
        t = make_tensor((4, 6))
        r = t.view(2, 3, 4)
        assert r.shape == (2, 3, 4)


class TestSqueeze:
    def test_squeeze_dim_none(self):
        t = make_tensor((1, 3, 1, 4))
        r = lucid.squeeze(t)
        assert r.shape == (3, 4)

    def test_squeeze_specific_dim(self):
        t = make_tensor((1, 3, 4))
        r = lucid.squeeze(t, 0)
        assert r.shape == (3, 4)

    def test_squeeze_list(self):
        t = make_tensor((1, 3, 1, 4))
        r = lucid.squeeze(t, [0, 2])
        assert r.shape == (3, 4)

    def test_squeeze_non_unit_dim_noop(self):
        t = make_tensor((2, 3))
        r = lucid.squeeze(t, 0)  # dim 0 has size 2 — noop
        assert r.shape == (2, 3)


class TestUnsqueeze:
    def test_unsqueeze_front(self):
        t = make_tensor((3, 4))
        r = lucid.unsqueeze(t, 0)
        assert r.shape == (1, 3, 4)

    def test_unsqueeze_back(self):
        t = make_tensor((3, 4))
        r = lucid.unsqueeze(t, -1)
        assert r.shape == (3, 4, 1)

    def test_unsqueeze_mid(self):
        t = make_tensor((3, 4))
        r = lucid.unsqueeze(t, 1)
        assert r.shape == (3, 1, 4)


class TestPermute:
    def test_permute(self):
        t = make_tensor((2, 3, 4))
        r = t.permute(2, 0, 1)
        assert r.shape == (4, 2, 3)

    def test_transpose_2d(self):
        t = make_tensor((3, 4))
        r = lucid.transpose(t)
        assert r.shape == (4, 3)

    def test_swapaxes(self):
        t = make_tensor((2, 3, 4))
        r = t.swapaxes(0, 2)
        assert r.shape == (4, 3, 2)

    def test_mT(self):
        t = make_tensor((2, 3, 4))
        r = t.mT
        assert r.shape == (2, 4, 3)

    def test_T_property_2d(self):
        t = make_tensor((3, 4))
        assert t.T.shape == (4, 3)


class TestFlatten:
    def test_flatten_all(self):
        t = make_tensor((2, 3, 4))
        assert t.flatten().shape == (24,)

    def test_flatten_partial(self):
        t = make_tensor((2, 3, 4))
        r = t.flatten(1)
        assert r.shape == (2, 12)

    def test_ravel(self):
        t = make_tensor((2, 3))
        assert lucid.ravel(t).shape == (6,)


class TestExpand:
    def test_broadcast_to(self):
        t = make_tensor((1, 4))
        r = lucid.broadcast_to(t, (3, 4))
        assert r.shape == (3, 4)

    def test_expand(self):
        t = make_tensor((3, 1))
        r = t.expand(3, 4)
        assert r.shape == (3, 4)


class TestConcatStack:
    def test_cat_axis0(self):
        a = make_tensor((2, 3))
        b = make_tensor((4, 3))
        r = lucid.cat([a, b], 0)
        assert r.shape == (6, 3)

    def test_cat_axis1(self):
        a = make_tensor((3, 2))
        b = make_tensor((3, 5))
        r = lucid.cat([a, b], 1)
        assert r.shape == (3, 7)

    def test_stack(self):
        a = make_tensor((3, 4))
        b = make_tensor((3, 4))
        r = lucid.stack([a, b], 0)
        assert r.shape == (2, 3, 4)

    def test_hstack(self):
        a = make_tensor((3, 2))
        b = make_tensor((3, 4))
        r = lucid.hstack([a, b])
        assert r.shape == (3, 6)

    def test_vstack(self):
        a = make_tensor((2, 4))
        b = make_tensor((3, 4))
        r = lucid.vstack([a, b])
        assert r.shape == (5, 4)


class TestSplit:
    def test_split_equal_chunks(self):
        t = make_tensor((6, 4))
        parts = lucid.split(t, 2, 0)  # chunk_size=2 → 3 parts
        assert len(parts) == 3
        for p in parts:
            assert p.shape == (2, 4)

    def test_split_list_sections(self):
        t = make_tensor((6, 4))
        parts = lucid.split(t, [2, 3, 1], 0)
        assert [p.shape[0] for p in parts] == [2, 3, 1]

    def test_chunk(self):
        t = make_tensor((8, 3))
        parts = lucid.chunk(t, 4, 0)
        assert len(parts) == 4


class TestRepeat:
    def test_tile(self):
        t = lucid.tensor([[1.0, 2.0]])
        r = lucid.tile(t, [2, 3])
        assert r.shape == (2, 6)

    def test_tensor_repeat_tiles(self):
        t = lucid.tensor([[1.0, 2.0]])
        r = t.repeat(2, 3)
        assert r.shape == (2, 6)

    def test_repeat_interleave(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        r = lucid.repeat_interleave(t, 2, dim=0)
        expected = lucid.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        assert_close(r, expected)


class TestTriangular:
    def test_tril(self):
        t = lucid.ones(3, 3)
        r = lucid.tril(t)
        arr = r.numpy()
        assert arr[0, 1] == 0.0
        assert arr[1, 0] == 1.0

    def test_triu(self):
        t = lucid.ones(3, 3)
        r = lucid.triu(t)
        arr = r.numpy()
        assert arr[1, 0] == 0.0
        assert arr[0, 1] == 1.0
