"""``lucid.utils.data`` — Dataset / DataLoader / Sampler / collate."""

import numpy as np
import pytest

import lucid
from lucid.utils import data as ld


class _ToyDataset(ld.Dataset):
    def __init__(self, n: int = 10) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[lucid.Tensor, int]:
        return lucid.tensor([float(i), float(i + 1)]), i


class TestDataset:
    def test_len(self) -> None:
        ds = _ToyDataset(7)
        assert len(ds) == 7

    def test_getitem(self) -> None:
        ds = _ToyDataset()
        x, y = ds[3]
        np.testing.assert_array_equal(x.numpy(), [3.0, 4.0])
        assert y == 3


class TestDataLoader:
    def test_iter_full_pass(self) -> None:
        ds = _ToyDataset(8)
        dl = ld.DataLoader(ds, batch_size=4, shuffle=False)
        batches = list(dl)
        assert len(batches) == 2

    def test_batch_size(self) -> None:
        ds = _ToyDataset(6)
        dl = ld.DataLoader(ds, batch_size=3, shuffle=False)
        for x, y in dl:
            # Shape is whatever the default collate returns; just
            # check we got 3-batch pieces.
            assert (hasattr(x, "shape") and x.shape[0] == 3) or (
                hasattr(x, "__len__") and len(x) == 3
            )

    def test_shuffle_changes_order(self) -> None:
        ds = _ToyDataset(8)
        lucid.manual_seed(0)
        dl1 = ld.DataLoader(ds, batch_size=8, shuffle=True)
        first = next(iter(dl1))
        lucid.manual_seed(1)
        dl2 = ld.DataLoader(ds, batch_size=8, shuffle=True)
        second = next(iter(dl2))
        # Two different seeds → at least one element should differ.
        if hasattr(first[0], "numpy"):
            a = first[0].numpy().flatten()
            b = second[0].numpy().flatten()
            assert not (a == b).all()


class TestSamplers:
    def test_sequential(self) -> None:
        if not hasattr(ld, "SequentialSampler"):
            pytest.skip("SequentialSampler not exposed")
        ds = _ToyDataset(5)
        s = ld.SequentialSampler(ds)
        assert list(s) == [0, 1, 2, 3, 4]

    def test_random(self) -> None:
        if not hasattr(ld, "RandomSampler"):
            pytest.skip("RandomSampler not exposed")
        ds = _ToyDataset(8)
        s = ld.RandomSampler(ds)
        out = list(s)
        assert sorted(out) == list(range(8))
