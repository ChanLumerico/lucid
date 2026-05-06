"""
Dataset base classes and implementations.
"""

from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class Dataset:
    """Abstract base class for datasets. Subclasses must implement __len__ and __getitem__."""

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __add__(self, other: Dataset) -> ConcatDataset:
        return ConcatDataset([self, other])


class IterableDataset:
    """Base class for iterable-style datasets. Subclasses must implement __iter__."""

    def __iter__(self) -> Iterator[Tensor | tuple[Tensor, ...]]:
        raise NotImplementedError

    def __add__(self, other: IterableDataset) -> IterableDataset:
        raise NotImplementedError("Concatenation of IterableDatasets is not supported.")


class TensorDataset(Dataset):
    """Dataset wrapping Tensors. Each sample is a tuple of slices along the first dim."""

    def __init__(self, *tensors: Tensor) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        n = tensors[0].shape[0]
        for t in tensors[1:]:
            if t.shape[0] != n:
                raise ValueError(
                    f"All tensors must have the same size in the first dimension: "
                    f"expected {n}, got {t.shape[0]}"
                )
        self.tensors = tensors

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(t[index] for t in self.tensors)

    def __len__(self) -> int:
        return self.tensors[0].shape[0]


class ConcatDataset(Dataset):
    """Dataset that concatenates multiple datasets."""

    def __init__(self, datasets: list[Dataset]) -> None:
        self.datasets = list(datasets)
        self.cumulative_sizes = self._cumsum([len(d) for d in datasets])

    @staticmethod
    def _cumsum(sizes: list[int]) -> list[int]:
        result = []
        total = 0
        for s in sizes:
            total += s
            result.append(total)
        return result

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, ...]:
        if idx < 0:
            idx = len(self) + idx
        dataset_idx = 0
        while (
            dataset_idx < len(self.cumulative_sizes)
            and idx >= self.cumulative_sizes[dataset_idx]
        ):
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class Subset(Dataset):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tensor | tuple[Tensor, ...]:
        return self.dataset[self.indices[idx]]


def random_split(
    dataset: Dataset,
    lengths: list[int] | list[float],
    generator: object = None,
) -> list[Subset]:
    """Randomly split a dataset into non-overlapping subsets.

    Args:
        dataset:   Dataset to split.
        lengths:   Lengths of each split (ints) or fractions summing to 1 (floats).
        generator: Optional random generator for reproducibility.

    Returns:
        List of Subsets.
    """
    import random as _random

    n = len(dataset)

    if all(isinstance(x, float) for x in lengths):
        total = sum(lengths)  # type: ignore[arg-type]
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Fraction lengths must sum to 1.0")
        int_lengths = [round(x * n) for x in lengths]  # type: ignore[arg-type]
        # Fix rounding error on last element
        int_lengths[-1] = n - sum(int_lengths[:-1])
        lengths = int_lengths

    lengths_int: list[int] = [int(x) for x in lengths]
    if sum(lengths_int) != n:
        raise ValueError(
            f"Sum of split lengths ({sum(lengths_int)}) must equal dataset length ({n})"
        )

    indices = list(range(n))
    if generator is not None:
        rng = _random.Random(generator)
        rng.shuffle(indices)
    else:
        _random.shuffle(indices)

    result = []
    offset = 0
    for length in lengths_int:
        result.append(Subset(dataset, indices[offset : offset + length]))
        offset += length
    return result


class ChainDataset(IterableDataset):
    """Concatenate multiple :class:`IterableDataset` instances end-to-end.

    Iteration yields every element from the first dataset, then every element
    from the second, and so on. Mirrors ``reference framework.utils.data.ChainDataset``.

    Each child must be an :class:`IterableDataset`; map-style datasets must be
    iterated through a sampler-driven DataLoader instead.
    """

    def __init__(self, datasets: "list[IterableDataset]") -> None:
        bad: list[type] = [
            type(d) for d in datasets if not isinstance(d, IterableDataset)
        ]
        if bad:
            raise TypeError(
                f"ChainDataset requires IterableDataset children, got {bad}"
            )
        self.datasets: list[IterableDataset] = list(datasets)

    def __iter__(self) -> "Iterator[Tensor | tuple[Tensor, ...]]":
        for d in self.datasets:
            yield from d


class StackDataset(Dataset):
    """Bundle several map-style datasets so each index returns a stacked tuple.

    The bundled datasets must agree in length. Item ``i`` is the tuple of
    ``d[i]`` for each child dataset — handy when paired modalities (image,
    caption, label) live in separate sources but share an index.

    Mirrors ``reference framework.utils.data.StackDataset``. Accepts both positional
    children (``StackDataset(d1, d2)``) and keyword children
    (``StackDataset(image=d1, label=d2)``); positional and keyword forms are
    mutually exclusive — match the reference framework's behaviour.
    """

    def __init__(self, *args: Dataset, **kwargs: Dataset) -> None:
        if args and kwargs:
            raise ValueError(
                "StackDataset takes either positional or keyword child datasets, not both"
            )
        children: tuple[Dataset, ...] = args if args else tuple(kwargs.values())
        if not children:
            raise ValueError("StackDataset requires at least one child dataset")
        n: int = len(children[0])
        for child in children[1:]:
            if len(child) != n:
                raise ValueError(
                    f"StackDataset children must agree in length; "
                    f"saw {n} then {len(child)}"
                )
        self.datasets: tuple[Dataset, ...] = children
        self._keys: tuple[str, ...] | None = tuple(kwargs.keys()) if kwargs else None
        self._n: int = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[object, ...] | dict[str, object]:
        items: tuple[object, ...] = tuple(d[idx] for d in self.datasets)
        if self._keys is not None:
            return dict(zip(self._keys, items))
        return items
