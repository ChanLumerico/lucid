import numpy as np

import pytest

import lucid

from lucid.data import DataLoader, Subset, TensorDataset, random_split


def _make_ds(n: int = 20):
    x = lucid.tensor(np.arange(n).reshape(n, 1).astype(np.float64))
    y = lucid.tensor(np.arange(n).astype(np.int64))
    return TensorDataset(x, y)


def test_tensor_dataset_indexing():
    ds = _make_ds(10)
    assert len(ds) == 10
    sample = ds[3]
    assert isinstance(sample, tuple) and len(sample) == 2
    np.testing.assert_array_equal(np.asarray(sample[0].data), np.array([3.0]))


def test_dataloader_no_shuffle_covers_all():
    ds = _make_ds(20)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    seen = []
    for batch in loader:
        x_batch, _ = batch
        seen.extend(np.asarray(x_batch.data).flatten().tolist())
    assert sorted(seen) == list(range(20))
    assert seen == list(range(20))


def test_dataloader_shuffle_covers_all_but_reorders():
    ds = _make_ds(20)
    lucid.random.seed(123)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    seen = []
    for batch in loader:
        x_batch, _ = batch
        seen.extend(np.asarray(x_batch.data).flatten().tolist())
    assert sorted(seen) == list(range(20))
    assert seen != list(range(20))


def test_dataloader_batch_count():
    ds = _make_ds(10)
    loader = DataLoader(ds, batch_size=3, shuffle=False)
    batches = list(loader)
    assert len(batches) == 4
    assert batches[-1][0].shape[0] == 1


def test_random_split_int_lengths_partition():
    ds = _make_ds(10)
    a, b = random_split(ds, [7, 3], seed=42)
    assert isinstance(a, Subset) and isinstance(b, Subset)
    a_idx = set(a.indices)
    b_idx = set(b.indices)
    assert a_idx & b_idx == set()
    assert a_idx | b_idx == set(range(10))


def test_random_split_fractions_partition():
    ds = _make_ds(20)
    a, b, c = random_split(ds, [0.5, 0.3, 0.2], seed=1)
    total = sum((len(s) for s in (a, b, c)))
    assert total == 20
    all_idx = set(a.indices) | set(b.indices) | set(c.indices)
    assert all_idx == set(range(20))


def test_random_split_deterministic_with_seed():
    ds = _make_ds(15)
    a1, b1 = random_split(ds, [10, 5], seed=7)
    a2, b2 = random_split(ds, [10, 5], seed=7)
    assert list(a1.indices) == list(a2.indices)
    assert list(b1.indices) == list(b2.indices)


def test_random_split_raises_on_mixed_types():
    ds = _make_ds(10)
    with pytest.raises(TypeError, match="lengths must be all integers or all floats"):
        random_split(ds, [5, 0.5])


def test_random_split_rejects_wrong_int_sum():
    ds = _make_ds(10)
    with pytest.raises(ValueError, match="does not equal dataset length"):
        random_split(ds, [3, 4])


def test_random_split_rejects_negative():
    ds = _make_ds(10)
    with pytest.raises(ValueError, match="non-negative"):
        random_split(ds, [-1, 11])


def test_random_split_fractions_must_sum_to_one():
    ds = _make_ds(10)
    with pytest.raises(ValueError, match="must sum to 1"):
        random_split(ds, [0.3, 0.3])
