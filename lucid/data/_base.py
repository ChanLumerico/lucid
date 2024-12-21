from abc import ABC, abstractmethod
from typing import Self, Any
import random

import lucid


class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index: int) -> None:
        raise NotImplementedError("Subclasses must implement __getitem__.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("Subclasses must implement __len__.")

    def __add__(self, other: Self) -> Self:
        return ConcatDataset([self, other])


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

    def __getitem__(self, idx: int) -> Any:
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
        self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
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

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        start = self.current_index
        end = min(start + self.batch_size, len(self.indices))

        batch_indices = self.indices[start:end]
        batch = self.dataset[batch_indices]

        self.current_index = end
        return batch
