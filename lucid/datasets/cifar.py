import pandas as pd
import numpy as np
import openml
import math

from typing import SupportsIndex, Tuple, ClassVar

import lucid
from lucid._tensor import Tensor

from ._base import DatasetBase


__all__ = ["CIFAR10", "CIFAR100"]


class CIFAR10(DatasetBase):
    OPENML_ID: ClassVar[int] = 40927

    def _download(self) -> None:
        try:
            dataset = openml.datasets.get_dataset(self.OPENML_ID)
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            df.to_csv(self.root / "CIFAR10.csv", index=False)

        except Exception as e:
            raise RuntimeError(f"Failed to download the CIFAR-10 dataset. Error: {e}")

    def _load_data(self, split: str) -> Tuple[Tensor, Tensor]:
        csv_path = self.root / "CIFAR10.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"CIFAR-10 dataset CSV file not found at {csv_path}. "
                + "Use `download=True`."
            )

        df = pd.read_csv(csv_path)
        labels = df["class"].values.astype(np.int32)
        images = df.drop(columns=["class"]).values.astype(np.float32)
        images = images.reshape(-1, 3, 32, 32)

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
        image = self.data[index].reshape(-1, 3, 32, 32)
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class CIFAR100(DatasetBase):
    OPENML_ID: ClassVar[int] = 41983

    def _download(self) -> None:
        try:
            dataset = openml.datasets.get_dataset(self.OPENML_ID)
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            df.to_csv(self.root / "CIFAR100.csv", index=False)

        except Exception as e:
            raise RuntimeError(f"Failed to download the CIFAR-100 dataset. Error: {e}")

    def _load_data(self, split: str) -> Tuple[Tensor, Tensor]:
        csv_path = self.root / "CIFAR100.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"CIFAR-100 dataset CSV file not found at {csv_path}. "
                + "Use `download=True`."
            )

        df = pd.read_csv(csv_path)
        labels = df["class"].values.astype(np.int32)
        images = df.drop(columns=["class"]).values.astype(np.float32)
        images = images.reshape(-1, 3, 32, 32)

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
        image = self.data[index].reshape(-1, 3, 32, 32)
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
