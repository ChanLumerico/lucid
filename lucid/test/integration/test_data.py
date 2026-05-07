"""Tests for ``lucid.utils.data`` dataset/sampler/loader plumbing."""

import pytest

import lucid
from lucid.utils.data import (
    ChainDataset,
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    StackDataset,
    Subset,
    TensorDataset,
    random_split,
)


class _Range(IterableDataset):
    """Trivial iterable dataset used as fodder for ChainDataset tests."""

    def __init__(self, start: int, stop: int) -> None:
        self.start: int = start
        self.stop: int = stop

    def __iter__(self):  # type: ignore[no-untyped-def]
        for i in range(self.start, self.stop):
            yield i


class TestChainDataset:
    def test_iterates_in_order(self) -> None:
        chain: ChainDataset = ChainDataset([_Range(0, 3), _Range(10, 12)])
        assert list(chain) == [0, 1, 2, 10, 11]

    def test_rejects_map_style_child(self) -> None:
        td: TensorDataset = TensorDataset(lucid.tensor([[1.0]]))
        with pytest.raises(TypeError, match="IterableDataset"):
            ChainDataset([td])  # type: ignore[list-item]


class TestStackDataset:
    def _pair(self) -> tuple[TensorDataset, TensorDataset]:
        d1: TensorDataset = TensorDataset(lucid.tensor([[1.0, 2.0], [3.0, 4.0]]))
        d2: TensorDataset = TensorDataset(lucid.tensor([[10.0], [20.0]]))
        return d1, d2

    def test_positional_returns_tuple(self) -> None:
        d1, d2 = self._pair()
        stacked: StackDataset = StackDataset(d1, d2)
        assert len(stacked) == 2
        item = stacked[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_keyword_returns_dict(self) -> None:
        d1, d2 = self._pair()
        stacked: StackDataset = StackDataset(image=d1, label=d2)
        item = stacked[0]
        assert isinstance(item, dict)
        assert set(item.keys()) == {"image", "label"}

    def test_length_mismatch_raises(self) -> None:
        d1, _ = self._pair()
        d3: TensorDataset = TensorDataset(lucid.tensor([[1.0], [2.0], [3.0]]))
        with pytest.raises(ValueError, match="agree in length"):
            StackDataset(d1, d3)

    def test_mixed_positional_and_keyword_rejected(self) -> None:
        d1, d2 = self._pair()
        with pytest.raises(ValueError, match="not both"):
            StackDataset(d1, label=d2)

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            StackDataset()


class TestExistingDatasetsStillWork:
    """A regression sanity check that the new dataset additions did not
    disturb the existing surface."""

    def test_concat_and_subset_loader(self) -> None:
        d1: TensorDataset = TensorDataset(lucid.tensor([[1.0], [2.0]]))
        d2: TensorDataset = TensorDataset(lucid.tensor([[3.0], [4.0]]))
        merged: ConcatDataset = ConcatDataset([d1, d2])
        sub: Subset = Subset(merged, [0, 2, 3])
        loader: DataLoader = DataLoader(sub, batch_size=2, shuffle=False)
        batches: list[object] = list(loader)
        assert len(batches) == 2

    def test_random_split_preserves_total(self) -> None:
        td: TensorDataset = TensorDataset(lucid.arange(10).reshape(10, 1))
        a, b = random_split(td, [7, 3])
        assert len(a) + len(b) == 10


class TestDistributedSampler:
    def _make(self) -> TensorDataset:
        return TensorDataset(lucid.arange(10).reshape(10, 1))

    def test_single_rank_yields_all_indices(self) -> None:
        from lucid.utils.data import DistributedSampler

        sampler = DistributedSampler(
            self._make(), num_replicas=1, rank=0, shuffle=False
        )
        assert list(sampler) == list(range(10))
        assert len(sampler) == 10

    def test_two_replicas_split_disjointly(self) -> None:
        from lucid.utils.data import DistributedSampler

        td: TensorDataset = self._make()
        rank0 = list(DistributedSampler(td, num_replicas=2, rank=0, shuffle=False))
        rank1 = list(DistributedSampler(td, num_replicas=2, rank=1, shuffle=False))
        assert rank0 == [0, 2, 4, 6, 8]
        assert rank1 == [1, 3, 5, 7, 9]
        assert set(rank0) | set(rank1) == set(range(10))

    def test_epoch_affects_shuffle(self) -> None:
        from lucid.utils.data import DistributedSampler

        sampler = DistributedSampler(
            self._make(), num_replicas=1, rank=0, shuffle=True, seed=42
        )
        sampler.set_epoch(0)
        first: list[int] = list(sampler)
        sampler.set_epoch(1)
        second: list[int] = list(sampler)
        assert first != second
        assert sorted(first) == sorted(second)  # same elements, different order

    def test_invalid_rank_rejected(self) -> None:
        import pytest as _pytest
        from lucid.utils.data import DistributedSampler

        with _pytest.raises(ValueError, match="rank"):
            DistributedSampler(self._make(), num_replicas=2, rank=2)
