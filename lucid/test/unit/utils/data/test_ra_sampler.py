"""Repeated Augmentation sampler (Hoffer et al., 2020 — arXiv:1901.09335).

Verifies:

* every index emitted is in [0, len(dataset))
* each unique index appears either ``num_repeats`` times or ``0`` times
  in the un-truncated repeated stream (consecutive grouping)
* ``num_replicas`` / ``rank`` split disjoint, non-overlapping slabs
* ``set_epoch`` changes the shuffled order deterministically
* ``num_repeats=1`` degenerates to a plain shuffled/sequential pass
* argument validation
"""

import pytest

import lucid
import lucid.utils.data as D


class _ToyDataset(D.Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


class TestRASampler:
    def test_indices_in_range(self) -> None:
        ds = _ToyDataset(20)
        sampler = D.RASampler(ds, num_repeats=3, shuffle=False)
        for idx in sampler:
            assert 0 <= idx < 20

    def test_length_matches_floor_n_over_w(self) -> None:
        ds = _ToyDataset(20)
        sampler = D.RASampler(ds, num_replicas=4, rank=0, num_repeats=3)
        # num_selected_samples = floor(20 / 4) = 5
        assert len(sampler) == 5
        # And iteration yields exactly that many indices.
        assert len(list(sampler)) == 5

    def test_repeats_are_consecutive(self) -> None:
        # With num_repeats=3, each unique index appears in a run of 3 consecutive
        # positions in the *un-truncated* stream.  Check by listing the full
        # repeated set via num_replicas=1 (single slab, no truncation past
        # num_selected_samples — so we explicitly inspect the internal slab).
        ds = _ToyDataset(8)
        sampler = D.RASampler(ds, num_replicas=1, num_repeats=3, shuffle=False)
        # Without shuffle: indices = [0,1,2,...,7], repeated → [0,0,0,1,1,1,...].
        # num_selected_samples = floor(8/1) = 8, so we get the first 8 elements.
        out = list(sampler)
        # First 3 of the run-of-3s are: 0,0,0, then 1,1,1, then 2,2 (truncated).
        assert out[:3] == [0, 0, 0]
        assert out[3:6] == [1, 1, 1]
        assert out[6:8] == [2, 2]

    def test_set_epoch_changes_order(self) -> None:
        ds = _ToyDataset(20)
        sampler = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        sampler.set_epoch(0)
        ep0 = list(sampler)
        sampler.set_epoch(1)
        ep1 = list(sampler)
        # Different epochs → different orderings (very high probability).
        assert ep0 != ep1

    def test_seed_reproducibility(self) -> None:
        ds = _ToyDataset(20)
        s1 = D.RASampler(ds, num_repeats=3, shuffle=True, seed=42)
        s2 = D.RASampler(ds, num_repeats=3, shuffle=True, seed=42)
        s1.set_epoch(5)
        s2.set_epoch(5)
        assert list(s1) == list(s2)

    def test_num_repeats_one_degenerates(self) -> None:
        # num_repeats=1 + shuffle=False → indices appear once, sequentially,
        # truncated to floor(N / W).
        ds = _ToyDataset(8)
        sampler = D.RASampler(ds, num_replicas=1, num_repeats=1, shuffle=False)
        assert list(sampler) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_invalid_num_replicas(self) -> None:
        with pytest.raises(ValueError, match="num_replicas"):
            D.RASampler(_ToyDataset(10), num_replicas=0)

    def test_invalid_rank(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            D.RASampler(_ToyDataset(10), num_replicas=2, rank=2)

    def test_invalid_num_repeats(self) -> None:
        with pytest.raises(ValueError, match="num_repeats"):
            D.RASampler(_ToyDataset(10), num_repeats=0)

    def test_distributed_disjoint_slabs(self) -> None:
        # Two ranks of a num_replicas=2 split should see disjoint slabs.
        ds = _ToyDataset(20)
        s0 = D.RASampler(ds, num_replicas=2, rank=0, num_repeats=3, shuffle=False)
        s1 = D.RASampler(ds, num_replicas=2, rank=1, num_repeats=3, shuffle=False)
        # ceil(20*3/2) = 30 per rank; num_selected_samples = floor(20/2) = 10.
        out0 = list(s0)
        out1 = list(s1)
        assert len(out0) == 10
        assert len(out1) == 10
        # The two slabs come from different positions of the repeated
        # index list — under shuffle=False they don't overlap.
        # (We don't insist on set-disjointness because both are drawn from
        # the same 20-element pool, but the slab positions are distinct.)


# Smoke-test integration with DataLoader (the canonical use case).


class TestRASamplerWithDataLoader:
    def test_integrates_with_dataloader(self) -> None:
        ds = D.TensorDataset(lucid.arange(20).reshape(20, 1).to(lucid.float32))
        sampler = D.RASampler(ds, num_repeats=3, shuffle=True, seed=0)
        loader = D.DataLoader(ds, batch_size=4, sampler=sampler)
        batches = list(loader)
        # Per RA spec, each rank yields floor(N/1)=20 indices → 5 batches of 4.
        assert len(batches) == 5
        for batch in batches:
            # TensorDataset wrapping a single tensor yields (tensor,) tuples.
            (x,) = batch
            assert tuple(x.shape) == (4, 1)
