"""Parity: ``lucid.utils.data`` vs reference framework data utilities."""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.utils.data import (
    Dataset,
    Subset,
    random_split,
    default_convert,
    collate,
)
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestDefaultConvertParity:
    def test_ndarray_to_tensor(self, ref: Any) -> None:  # noqa: ARG002
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        l = default_convert(arr)
        r = ref.utils.data.dataloader.default_convert(arr.copy())
        assert isinstance(l, lucid.Tensor)
        assert_close(l, r, atol=0.0)

    def test_scalar_numpy_int(self, ref: Any) -> None:  # noqa: ARG002
        import numpy as np

        val = np.int64(42)
        l = default_convert(val)
        r = ref.utils.data.dataloader.default_convert(val)
        assert int(l.item()) == int(r.item())

    def test_list_passthrough(self, ref: Any) -> None:  # noqa: ARG002
        lst = ["a", "b", "c"]
        l = default_convert(lst)
        r = ref.utils.data.dataloader.default_convert(lst)
        assert l == r


@pytest.mark.parity
class TestCollateParity:
    def test_list_of_arrays(self, ref: Any) -> None:
        np.random.seed(0)
        batch = [np.random.standard_normal(4).astype(np.float32) for _ in range(6)]
        l = collate(batch)
        r = ref.utils.data.dataloader.default_collate(
            [ref.tensor(b.copy()) for b in batch]
        )
        assert_close(l, r, atol=1e-6)

    def test_list_of_tuples(self, ref: Any) -> None:
        # Use float arrays for both elements to avoid scalar-collation differences.
        np.random.seed(1)
        xs = [np.random.standard_normal(3).astype(np.float32) for _ in range(4)]
        ys = [np.array([float(i)], dtype=np.float32) for i in range(4)]
        batch = [(xs[i], ys[i]) for i in range(4)]

        l_x, l_y = collate(batch)
        r_out = ref.utils.data.dataloader.default_collate(
            [(ref.tensor(xs[i].copy()), ref.tensor(ys[i].copy())) for i in range(4)]
        )
        r_x, r_y = r_out

        assert_close(l_x, r_x, atol=1e-6)
        assert_close(l_y, r_y, atol=1e-6)


@pytest.mark.parity
class TestSubsetParity:
    def _make_dataset(self, n: int = 10) -> Dataset:
        class _DS(Dataset):
            def __len__(self) -> int:
                return n

            def __getitem__(self, idx: int) -> lucid.Tensor:
                return lucid.tensor([float(idx)])

        return _DS()

    def test_subset_len(self, ref: Any) -> None:  # noqa: ARG002
        ds = self._make_dataset(10)
        sub = Subset(ds, [0, 2, 4, 6])
        assert len(sub) == 4

    def test_subset_getitem(self, ref: Any) -> None:  # noqa: ARG002
        ds = self._make_dataset(10)
        sub = Subset(ds, [3, 7])
        assert sub[0].item() == 3.0
        assert sub[1].item() == 7.0


@pytest.mark.parity
class TestRandomSplitParity:
    def _make_dataset(self, n: int = 20) -> Dataset:
        class _DS(Dataset):
            def __len__(self) -> int:
                return n

            def __getitem__(self, idx: int) -> lucid.Tensor:
                return lucid.tensor([float(idx)])

        return _DS()

    def test_split_sizes_sum(self, ref: Any) -> None:  # noqa: ARG002
        ds = self._make_dataset(20)
        a, b = random_split(ds, [12, 8])
        assert len(a) + len(b) == 20

    def test_split_no_overlap(self, ref: Any) -> None:  # noqa: ARG002
        ds = self._make_dataset(20)
        a, b = random_split(ds, [12, 8])
        idx_a = set(a.indices)
        idx_b = set(b.indices)
        assert idx_a.isdisjoint(idx_b)
        assert len(idx_a | idx_b) == 20
