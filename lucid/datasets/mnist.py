import pandas as pd
import numpy as np
import openml
import math

from typing import SupportsIndex, Tuple, ClassVar
from pathlib import Path
import re

import lucid
from lucid._tensor import Tensor

from ._base import DatasetBase


__all__ = ["MNIST", "FashionMNIST"]


class MNIST(DatasetBase):
    OPENML_ID: ClassVar[int] = 554

    def __init__(
        self,
        root: str | Path,
        train: bool | None = True,
        download: bool | None = False,
        transform: lucid.nn.Module | None = None,
        target_transform: lucid.nn.Module | None = None,
        test_size: float = 0.2,
        to_tensor: bool = True,
        *,
        cache: bool = True,
        scale: float | None = None,
        resize: tuple[int, int] | None = None,
        normalize: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        cache_preprocessed: bool = True,
        preprocess_dtype: lucid.Numeric = lucid.Float16,
        preprocess_chunk_size: int = 4096,
    ) -> None:
        self.cache = cache
        self.scale = scale
        self.resize = resize
        self.normalize = normalize
        self.cache_preprocessed = cache_preprocessed
        self.preprocess_dtype = preprocess_dtype
        self.preprocess_chunk_size = preprocess_chunk_size

        super().__init__(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
            test_size=test_size,
            to_tensor=to_tensor,
        )

    def _download(self) -> None:
        try:
            dataset = openml.datasets.get_dataset(self.OPENML_ID)
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            df.to_csv(self.root / "MNIST.csv", index=False)

        except Exception as e:
            raise RuntimeError(f"Failed to download the MNIST dataset. Error: {e}")

    def _cache_key(self) -> str:
        parts: list[str] = []
        if self.scale is not None:
            parts.append(f"s{self.scale:g}")
        if self.resize is not None:
            parts.append(f"r{self.resize[0]}x{self.resize[1]}")
        if self.normalize is not None:
            mean, std = self.normalize
            parts.append("m" + ",".join(f"{v:g}" for v in mean))
            parts.append("v" + ",".join(f"{v:g}" for v in std))
        if not parts:
            return "raw"
        key = "_".join(parts)
        return re.sub(r"[^a-zA-Z0-9_,.x-]+", "_", key)

    def _raw_cache_path(self) -> Path:
        return self.root / "MNIST_int16.npz"

    def _proc_cache_path(self) -> Path:
        dtype_name = str(self.preprocess_dtype)
        return self.root / f"MNIST_{self._cache_key()}_{dtype_name}.npz"

    def _ensure_raw_cache(self) -> tuple[np.ndarray, np.ndarray]:
        raw_path = self._raw_cache_path()
        if self.cache and raw_path.exists():
            with np.load(raw_path) as npz:
                images = npz["images"]
                labels = npz["labels"]
            return images, labels

        csv_path = self.root / "MNIST.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"MNIST dataset CSV file not found at {csv_path}. Use `download=True`."
            )

        df = pd.read_csv(csv_path)
        labels = df["class"].values.astype(np.int32)
        images = df.drop(columns=["class"]).values.astype(np.float32)
        images = images.reshape(-1, 1, 28, 28).astype(np.int16)

        if self.cache:
            np.savez_compressed(raw_path, images=images, labels=labels)

        return images, labels

    def _maybe_preprocess_and_cache(
        self, images_int16: np.ndarray, labels_int32: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.resize is None and self.scale is None and self.normalize is None:
            return images_int16.astype(np.float32), labels_int32

        proc_path = self._proc_cache_path()
        if self.cache and self.cache_preprocessed and proc_path.exists():
            with np.load(proc_path) as npz:
                images = npz["images"]
                labels = npz["labels"]
            return images, labels

        from lucid.transforms import Compose, Resize, Normalize

        class _Scale(lucid.nn.Module):
            def __init__(self, factor: float) -> None:
                super().__init__()
                self.factor = factor

            def forward(self, x: Tensor) -> Tensor:
                return x * self.factor

        transforms: list[lucid.nn.Module] = []
        if self.resize is not None:
            transforms.append(Resize(self.resize))
        if self.scale is not None:
            transforms.append(_Scale(self.scale))
        if self.normalize is not None:
            mean, std = self.normalize
            transforms.append(Normalize(mean=mean, std=std))

        transform = Compose(transforms)
        n = images_int16.shape[0]
        out_h, out_w = self.resize if self.resize is not None else (28, 28)

        out_dtype = np.float16 if self.preprocess_dtype == lucid.Float16 else np.float32
        out_images = np.empty((n, 1, out_h, out_w), dtype=out_dtype)

        for start in range(0, n, self.preprocess_chunk_size):
            end = min(start + self.preprocess_chunk_size, n)
            chunk = images_int16[start:end].astype(np.float32)
            x = lucid.to_tensor(chunk, dtype=lucid.Float32)
            x = transform(x)
            out_images[start:end] = x.numpy().astype(out_dtype, copy=False)

        if self.cache and self.cache_preprocessed:
            np.savez_compressed(proc_path, images=out_images, labels=labels_int32)

        return out_images, labels_int32

    def _load_data(self, split: str) -> Tuple[Tensor, Tensor]:
        images, labels = self._ensure_raw_cache()
        images, labels = self._maybe_preprocess_and_cache(images, labels)

        train_size = int(math.floor(len(images) * (1 - self.test_size)))
        if split == "train":
            images, labels = images[:train_size], labels[:train_size]
        else:
            images, labels = images[train_size:], labels[train_size:]

        if self.to_tensor:
            if images.dtype == np.float16 and self.preprocess_dtype == lucid.Float16:
                images = lucid.to_tensor(images, dtype=lucid.Float16)
            else:
                images = lucid.to_tensor(images, dtype=lucid.Float32)
            labels = lucid.to_tensor(labels, dtype=lucid.Int32)

        return images, labels

    def __getitem__(self, index: SupportsIndex) -> Tuple[Tensor, Tensor]:
        image = self.data[index]
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class FashionMNIST(DatasetBase):
    OPENML_ID: ClassVar[int] = 40996

    def _download(self) -> None:
        try:
            dataset = openml.datasets.get_dataset(self.OPENML_ID)
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            df.to_csv(self.root / "FashionMNIST.csv", index=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download the FashionMNIST dataset. Error: {e}"
            )

    def _load_data(self, split: str) -> Tuple[Tensor, Tensor]:
        csv_path = self.root / "FashionMNIST.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"FashionMNIST dataset CSV file not found at {csv_path}. "
                + "Use `download=True`."
            )

        df = pd.read_csv(csv_path)
        labels = df["class"].values.astype(np.int32)
        images = df.drop(columns=["class"]).values.astype(np.float32)
        images = images.reshape(-1, 1, 28, 28)

        train_size = int(math.floor(len(images) * (1 - self.test_size)))
        if split == "train":
            images, labels = images[:train_size], labels[:train_size]
        else:
            images, labels = images[train_size:], labels[train_size:]

        if self.to_tensor:
            images = lucid.to_tensor(images, dtype=lucid.Float32)
            labels = lucid.to_tensor(labels, dtype=lucid.Int32)

        return images, labels

    def __getitem__(self, index: SupportsIndex) -> Tuple[Tensor, Tensor]:
        image = self.data[index].reshape(-1, 1, 28, 28)
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
