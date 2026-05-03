"""
Tests for lucid.utils.data: Dataset, DataLoader, Sampler, default_collate.
"""

import pytest
import numpy as np
import lucid
from lucid.utils.data import (
    Dataset,
    TensorDataset,
    ConcatDataset,
    Subset,
    DataLoader,
    default_collate,
    random_split,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)


class SimpleDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


class TestDataset:
    def test_tensor_dataset_len(self):
        X = lucid.randn(100, 4)
        y = lucid.randn(100)
        ds = TensorDataset(X, y)
        assert len(ds) == 100

    def test_tensor_dataset_getitem(self):
        X = lucid.randn(10, 4)
        y = lucid.randn(10)
        ds = TensorDataset(X, y)
        xi, yi = ds[3]
        assert xi.shape == (4,)
        assert yi.shape == ()

    def test_tensor_dataset_shape_mismatch(self):
        with pytest.raises(ValueError):
            TensorDataset(lucid.randn(10, 4), lucid.randn(9))

    def test_concat_dataset(self):
        ds1 = SimpleDataset(5)
        ds2 = SimpleDataset(3)
        cat = ConcatDataset([ds1, ds2])
        assert len(cat) == 8
        assert cat[4] == 4  # last element of ds1 (0-indexed: ds1[4]=4)
        assert cat[5] == 0  # first element of ds2
        assert cat[7] == 2  # last element of ds2

    def test_subset(self):
        ds = SimpleDataset(10)
        sub = Subset(ds, [0, 2, 4])
        assert len(sub) == 3
        assert sub[0] == 0
        assert sub[1] == 2
        assert sub[2] == 4

    def test_random_split(self):
        ds = SimpleDataset(100)
        train, val = random_split(ds, [80, 20])
        assert len(train) == 80
        assert len(val) == 20

    def test_random_split_fractions(self):
        ds = SimpleDataset(100)
        train, val = random_split(ds, [0.8, 0.2])
        assert len(train) + len(val) == 100


class TestSamplers:
    def test_sequential_sampler(self):
        ds = SimpleDataset(5)
        sampler = SequentialSampler(ds)
        assert list(sampler) == [0, 1, 2, 3, 4]
        assert len(sampler) == 5

    def test_random_sampler(self):
        ds = SimpleDataset(10)
        sampler = RandomSampler(ds)
        indices = list(sampler)
        assert sorted(indices) == list(range(10))

    def test_batch_sampler_no_drop(self):
        ds = SimpleDataset(10)
        sampler = SequentialSampler(ds)
        bs = BatchSampler(sampler, batch_size=3, drop_last=False)
        batches = list(bs)
        assert len(batches) == 4  # 3+3+3+1
        assert batches[-1] == [9]

    def test_batch_sampler_drop_last(self):
        ds = SimpleDataset(10)
        sampler = SequentialSampler(ds)
        bs = BatchSampler(sampler, batch_size=3, drop_last=True)
        assert len(list(bs)) == 3


class TestDataLoader:
    def test_basic_iteration(self):
        X = lucid.randn(20, 4)
        y = lucid.randn(20)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=5)
        batches = list(loader)
        assert len(batches) == 4
        xb, yb = batches[0]
        assert xb.shape == (5, 4)
        assert yb.shape == (5,)

    def test_drop_last(self):
        ds = TensorDataset(lucid.randn(21, 4), lucid.randn(21))
        loader = DataLoader(ds, batch_size=5, drop_last=True)
        batches = list(loader)
        assert len(batches) == 4  # floor(21/5) = 4

    def test_shuffle_no_error(self):
        ds = TensorDataset(lucid.randn(20, 4), lucid.randn(20))
        loader = DataLoader(ds, batch_size=20, shuffle=True)
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0].shape == (20, 4)

    def test_len(self):
        ds = TensorDataset(lucid.randn(100, 4), lucid.randn(100))
        loader = DataLoader(ds, batch_size=16)
        assert len(loader) == 7  # ceil(100/16)

    def test_mutually_exclusive_shuffle_sampler(self):
        ds = TensorDataset(lucid.randn(10, 2), lucid.randn(10))
        from lucid.utils.data import RandomSampler
        with pytest.raises(ValueError):
            DataLoader(ds, shuffle=True, sampler=RandomSampler(ds))


class TestDefaultCollate:
    def test_collate_tensors(self):
        batch = [lucid.ones(3), lucid.ones(3)]
        result = default_collate(batch)
        assert result.shape == (2, 3)

    def test_collate_numpy(self):
        batch = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = default_collate(batch)
        assert result.shape == (2, 2)

    def test_collate_dict(self):
        batch = [{"x": lucid.ones(2), "y": 1}, {"x": lucid.ones(2), "y": 2}]
        result = default_collate(batch)
        assert result["x"].shape == (2, 2)

    def test_collate_tuples(self):
        batch = [(lucid.ones(2), lucid.zeros(2)), (lucid.ones(2), lucid.zeros(2))]
        result = default_collate(batch)
        assert isinstance(result, tuple)
        assert result[0].shape == (2, 2)
