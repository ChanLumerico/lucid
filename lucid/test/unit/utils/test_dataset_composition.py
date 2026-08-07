"""The dataset and sampler types, checked on what they promise.

``utils/data/dataset.py`` sat at 37%: ``ConcatDataset``, ``ChainDataset``,
``StackDataset``, ``Subset`` and most of the sampler family were exported
and never constructed.  These are small, total functions over indices —
the kind where a wrong boundary is invisible until a training run reads
the wrong sample.

Each composition is checked against what it claims: a concatenation has
to return the same items in the same order as reading its parts in turn,
a subset has to map its own index onto the parent's, and a sampler has to
be a permutation of the range rather than merely the right length.
"""

import numpy as np
import pytest

import lucid
from lucid.utils.data import (
    BatchSampler,
    ChainDataset,
    ConcatDataset,
    RandomSampler,
    SequentialSampler,
    StackDataset,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
)


def _tensor_dataset(n: int, offset: float = 0.0) -> TensorDataset:
    values = np.arange(n, dtype=np.float64) + offset
    return TensorDataset(lucid.tensor(values), lucid.tensor(values * 10))


def _item(dataset, i):
    return tuple(float(np.asarray(t.numpy()).ravel()[0]) for t in dataset[i])


# ── TensorDataset ─────────────────────────────────────────────────────────────


def test_tensor_dataset_pairs_its_tensors():
    ds = _tensor_dataset(5)
    assert len(ds) == 5
    assert _item(ds, 3) == (3.0, 30.0)


def test_tensor_dataset_refuses_ragged_tensors():
    with pytest.raises(Exception):
        TensorDataset(lucid.tensor(np.zeros(4)), lucid.tensor(np.zeros(5)))


# ── concatenation ─────────────────────────────────────────────────────────────


def test_concat_reads_its_parts_in_order():
    a, b = _tensor_dataset(3), _tensor_dataset(2, offset=100.0)
    joined = ConcatDataset([a, b])
    assert len(joined) == 5
    assert [_item(joined, i) for i in range(5)] == [
        _item(a, 0),
        _item(a, 1),
        _item(a, 2),
        _item(b, 0),
        _item(b, 1),
    ]


def test_concat_indexes_from_the_end_too():
    joined = ConcatDataset([_tensor_dataset(3), _tensor_dataset(2, offset=100.0)])
    assert _item(joined, -1) == _item(joined, 4)


def test_concat_refuses_an_index_past_the_end():
    joined = ConcatDataset([_tensor_dataset(3)])
    with pytest.raises((IndexError, ValueError)):
        joined[3]


def test_concat_of_a_concat_flattens_the_indexing():
    inner = ConcatDataset([_tensor_dataset(2), _tensor_dataset(2, offset=10.0)])
    outer = ConcatDataset([inner, _tensor_dataset(2, offset=100.0)])
    assert len(outer) == 6
    assert _item(outer, 5) == (101.0, 1010.0)


# ── subset ────────────────────────────────────────────────────────────────────


def test_subset_maps_onto_the_parent():
    parent = _tensor_dataset(10)
    picked = Subset(parent, [7, 2, 5])
    assert len(picked) == 3
    assert [_item(picked, i) for i in range(3)] == [
        _item(parent, 7),
        _item(parent, 2),
        _item(parent, 5),
    ]


def test_a_subset_of_a_subset_composes():
    parent = _tensor_dataset(10)
    once = Subset(parent, [1, 3, 5, 7])
    twice = Subset(once, [0, 3])
    assert [_item(twice, i) for i in range(2)] == [_item(parent, 1), _item(parent, 7)]


# ── stack ─────────────────────────────────────────────────────────────────────


def test_stack_dataset_zips_by_index():
    a, b = _tensor_dataset(4), _tensor_dataset(4, offset=100.0)
    stacked = StackDataset(a, b)
    assert len(stacked) == 4
    left, right = stacked[2]
    assert _item(a, 2) == tuple(float(np.asarray(t.numpy()).ravel()[0]) for t in left)
    assert _item(b, 2) == tuple(float(np.asarray(t.numpy()).ravel()[0]) for t in right)


# ── chaining iterables ────────────────────────────────────────────────────────


def test_chain_dataset_runs_through_each_in_turn():
    from lucid.utils.data import IterableDataset

    class Counting(IterableDataset):
        def __init__(self, start, stop):
            self.start, self.stop = start, stop

        def __iter__(self):
            return iter(range(self.start, self.stop))

    chained = ChainDataset([Counting(0, 3), Counting(10, 12)])
    assert list(chained) == [0, 1, 2, 10, 11]


# ── samplers ──────────────────────────────────────────────────────────────────


def test_sequential_sampler_is_the_range():
    assert list(SequentialSampler(_tensor_dataset(5))) == [0, 1, 2, 3, 4]


def test_random_sampler_is_a_permutation_not_just_a_length():
    """A sampler that returned the same index five times would have the
    right length and lose four fifths of the data."""
    drawn = list(RandomSampler(_tensor_dataset(20)))
    assert sorted(drawn) == list(range(20))


def test_random_sampler_with_replacement_may_repeat():
    drawn = list(RandomSampler(_tensor_dataset(10), replacement=True, num_samples=30))
    assert len(drawn) == 30
    assert all(0 <= i < 10 for i in drawn)


def test_subset_random_sampler_stays_inside_its_subset():
    indices = [3, 5, 9]
    drawn = list(SubsetRandomSampler(indices))
    assert sorted(drawn) == sorted(indices)


@pytest.mark.parametrize("drop_last", [False, True])
def test_batch_sampler_partitions_the_indices(drop_last):
    batches = list(BatchSampler(SequentialSampler(_tensor_dataset(10)), 3, drop_last))
    flat = [i for batch in batches for i in batch]
    if drop_last:
        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)
        assert flat == list(range(9))
    else:
        assert len(batches) == 4
        assert batches[-1] == [9]
        assert flat == list(range(10))


def test_batch_sampler_length_agrees_with_what_it_yields():
    sampler = BatchSampler(SequentialSampler(_tensor_dataset(10)), 3, False)
    assert len(sampler) == len(list(sampler))
