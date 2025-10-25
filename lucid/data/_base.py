from abc import ABC, abstractmethod
from typing import Callable, Iterator, Self, Any, override
import random
import math

import lucid
from lucid._tensor import Tensor
from lucid.types import _ArrayLike, _IndexLike, _DeviceType


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, idx: _IndexLike) -> None:
        raise NotImplementedError("Subclasses must implement __getitem__.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__.")

    def __add__(self, other: Self) -> Self:
        return ConcatDataset([self, other])

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"Dataset(n={len(self)})"


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: _IndexLike) -> Any:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)

    @override
    def __iter__(self) -> Iterator[Any]:
        for i in self.indices:
            yield self.dataset[i]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __repr__(self) -> str:
        return f"Subset(n={len(self)})"


class TensorDataset(Dataset):
    def __init__(self, *tensors_or_arrays: Tensor | _ArrayLike) -> None:
        super().__init__()
        if len(tensors_or_arrays) == 0:
            raise ValueError(
                "TensorDataset requires at least one tensor/array-like object."
            )
        try:
            self._tensors: tuple[Tensor, ...] = tuple(
                lucid._check_is_tensor(t) for t in tensors_or_arrays
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to convert array-like object(s) to tensor."
            ) from e

        n0 = len(self._tensors[0])
        for i, t in enumerate(self._tensors):
            if t.ndim == 0 or len(t) == 0:
                raise RuntimeError(
                    "All tensors must be at least 1D. "
                    f"Tensor at index {i} has no length."
                )
            if len(t) != n0:
                raise ValueError(
                    "All tensors must have the same length along dim 0: "
                    f"got {n0} and {len(t)} at index {i}."
                )

    def __len__(self) -> int:
        return len(self._tensors[0])

    @override
    def __getitem__(self, idx: _IndexLike | Tensor) -> tuple[Tensor, ...]:
        return tuple(t[idx] for t in self._tensors)

    def to(self, device: _DeviceType) -> Self:
        self._tensors = tuple(t.to(device) for t in self._tensors)
        return self

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        return self._tensors

    @override
    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        return super().__iter__()

    @override
    def __repr__(self) -> str:
        shapes = ", ".join(str(t.shape) for t in self._tensors)
        devices = {t.device for t in self._tensors}
        return f"TensorDataset(n={len(self)}, shapes=({shapes}), devices={devices})"


class ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self) -> list[int]:
        cum_sizes = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            cum_sizes.append(total)

        return cum_sizes

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: _IndexLike) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise IndexError("Index out of range.")
            idx = len(self) + idx

        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                dataset_idx = i
                if dataset_idx > 0:
                    idx -= self.cumulative_sizes[dataset_idx - 1]

                return self.datasets[dataset_idx][idx]

        raise IndexError("Index out of range.")


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: Callable | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or self.default_collate
        self.indices = list(range(len(dataset)))

        if shuffle:
            self._shuffle_indices()

    def _shuffle_indices(self) -> None:
        random.shuffle(self.indices)

    def __iter__(self) -> Self:
        self.current_index = 0
        if self.shuffle:
            self._shuffle_indices()
        return self

    def __next__(self) -> Any:
        if self.current_index >= len(self.indices):
            raise StopIteration

        start = self.current_index
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]
        self.current_index = end

        batch = [self.dataset[i] for i in batch_indices]
        return self.collate_fn(batch)

    def __len__(self) -> int:
        return int(math.ceil(len(self.dataset) / self.batch_size))

    @staticmethod
    def default_collate(batch: list[Any]) -> Any:
        if isinstance(batch[0], (tuple, list)):
            transposed = list(zip(*batch))
            return tuple(lucid.stack(tuple(x), axis=0) for x in transposed)

        elif isinstance(batch[0], Tensor):
            return lucid.stack(tuple(batch), axis=0)
        else:
            return batch
