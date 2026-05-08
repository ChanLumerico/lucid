"""Unit tests for the P2-C engine + composite additions:
``put`` / ``index_put`` / ``index_put_`` and the RNG-state surface.

The bitwise-shift / nextafter / polygamma cases live alongside their
sibling ops in ``test_ops_extras.py`` — this file is for the new
top-level functions that don't fit there.
"""

import numpy as np
import pytest

import lucid

# ── put (flat-index scatter) ──────────────────────────────────────────────


class TestPut:
    def test_basic_overwrite(self) -> None:
        x = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = lucid.tensor([0, 5, 2], dtype=lucid.int64)
        src = lucid.tensor([10.0, 20.0, 30.0])
        out = lucid.put(x, idx, src).numpy()
        # flat[0]=10, flat[5]=20, flat[2]=30 → [[10, 2, 30], [4, 5, 20]].
        np.testing.assert_array_equal(out, [[10.0, 2.0, 30.0], [4.0, 5.0, 20.0]])

    def test_accumulate(self) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        idx = lucid.tensor([0, 3], dtype=lucid.int64)
        src = lucid.tensor([10.0, 20.0])
        out = lucid.put(x, idx, src, accumulate=True).numpy()
        np.testing.assert_array_equal(out, [[11.0, 2.0], [3.0, 24.0]])

    def test_duplicate_indices_accumulate(self) -> None:
        x = lucid.zeros(4)
        idx = lucid.tensor([0, 0, 1], dtype=lucid.int64)
        src = lucid.tensor([1.0, 2.0, 3.0])
        out = lucid.put(x, idx, src, accumulate=True).numpy()
        np.testing.assert_array_equal(out, [3.0, 3.0, 0.0, 0.0])

    def test_preserves_shape(self) -> None:
        x = lucid.zeros(2, 3, 4)
        out = lucid.put(
            x,
            lucid.tensor([0, 23], dtype=lucid.int64),
            lucid.tensor([1.0, 2.0]),
        )
        assert out.shape == x.shape


# ── index_put / index_put_ ────────────────────────────────────────────────


class TestIndexPut:
    def test_2d_overwrite(self) -> None:
        x = lucid.zeros(3, 4)
        out = lucid.index_put(
            x,
            (
                lucid.tensor([0, 1], dtype=lucid.int64),
                lucid.tensor([2, 3], dtype=lucid.int64),
            ),
            lucid.tensor([10.0, 20.0]),
        ).numpy()
        np.testing.assert_array_equal(
            out,
            [
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_2d_accumulate(self) -> None:
        x = lucid.ones(2, 2)
        out = lucid.index_put(
            x,
            (
                lucid.tensor([0, 0, 1], dtype=lucid.int64),
                lucid.tensor([0, 0, 1], dtype=lucid.int64),
            ),
            lucid.tensor([5.0, 7.0, 3.0]),
            accumulate=True,
        ).numpy()
        np.testing.assert_array_equal(out, [[13.0, 1.0], [1.0, 4.0]])

    def test_3d(self) -> None:
        x = lucid.zeros(2, 3, 4)
        out = lucid.index_put(
            x,
            (
                lucid.tensor([0, 1], dtype=lucid.int64),
                lucid.tensor([1, 2], dtype=lucid.int64),
                lucid.tensor([3, 0], dtype=lucid.int64),
            ),
            lucid.tensor([100.0, 200.0]),
        ).numpy()
        assert out[0, 1, 3] == 100.0
        assert out[1, 2, 0] == 200.0

    def test_partial_indexing_rejected(self) -> None:
        x = lucid.zeros(3, 4)
        with pytest.raises(NotImplementedError):
            lucid.index_put(
                x,
                (lucid.tensor([0, 1], dtype=lucid.int64),),
                lucid.tensor([10.0, 20.0]),
            )

    def test_inplace_returns_same_tensor(self) -> None:
        x = lucid.zeros(3, 4)
        ret = lucid.index_put_(
            x,
            (
                lucid.tensor([0, 1], dtype=lucid.int64),
                lucid.tensor([2, 3], dtype=lucid.int64),
            ),
            lucid.tensor([7.0, 9.0]),
        )
        assert ret is x
        assert x.numpy()[0, 2] == 7.0
        assert x.numpy()[1, 3] == 9.0


# ── RNG state ─────────────────────────────────────────────────────────────


class TestRngState:
    def test_manual_seed_initial_seed_roundtrip(self) -> None:
        lucid.manual_seed(7)
        assert lucid.initial_seed() == 7

    def test_seed_returns_used_value(self) -> None:
        s = lucid.seed()
        assert s == lucid.initial_seed()
        # Different calls produce different seeds with overwhelming probability.
        s2 = lucid.seed()
        assert s != s2  # entropy from os.urandom

    def test_state_roundtrip_reproduces_samples(self) -> None:
        lucid.manual_seed(99)
        _ = lucid.rand(8)  # advance the counter.
        state = lucid.get_rng_state()
        sample1 = lucid.rand(5).numpy()
        lucid.set_rng_state(state)
        sample2 = lucid.rand(5).numpy()
        np.testing.assert_array_equal(sample1, sample2)

    def test_state_shape(self) -> None:
        # Length-2 [seed, counter].
        state = lucid.get_rng_state()
        assert state.shape == (2,)
        assert state.dtype == lucid.int64

    def test_set_rng_state_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            lucid.set_rng_state(lucid.tensor([1, 2, 3], dtype=lucid.int64))

    def test_generator_class_basic(self) -> None:
        g = lucid.Generator(seed=42)
        assert g.seed == 42
        assert g.counter == 0
        g.set_seed(100)
        assert g.seed == 100
        assert g.counter == 0  # counter resets on set_seed.

    def test_explicit_generator_seeds_independently(self) -> None:
        # Two generators with different seeds produce different streams.
        g1 = lucid.Generator(seed=1)
        g2 = lucid.Generator(seed=2)
        s1 = lucid.rand(5, generator=g1).numpy()
        s2 = lucid.rand(5, generator=g2).numpy()
        assert not np.allclose(s1, s2)
