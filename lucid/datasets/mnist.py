import pandas as pd
import numpy as np
import openml
import math

from pathlib import Path
from typing import Optional, SupportsIndex, Tuple, Union

import lucid
import lucid.nn as nn

from lucid.transforms import Compose
from lucid.data import Dataset
from lucid._tensor import Tensor


__all__ = ["MNIST"]


class MNIST(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: Optional[bool] = True,
        download: Optional[bool] = False,
        transform: Optional[nn.Module | Compose] = None,
        target_transform: Optional[nn.Module | Compose] = None,
        test_size: float = 0.2,
        to_tensor: bool = True,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.test_size = test_size
        self.to_tensor = to_tensor
        self.root.mkdir(parents=True, exist_ok=True)

        if download:
            self._download()

        if self.train:
            self.data, self.targets = self._load_data("train")
        else:
            self.data, self.targets = self._load_data("test")

    def _download(self) -> None:
        try:
            dataset = openml.datasets.get_dataset(554)
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            df.to_csv(self.root / "MNIST.csv", index=False)

        except Exception as e:
            raise RuntimeError(f"Failed to download the MNIST dataset. Error: {e}")

    def _load_data(self, split: str) -> Tuple[Tensor, Tensor]:
        csv_path = self.root / "MNIST.csv"
        if not csv_path.exists():
            raise RuntimeError(
                f"MNIST dataset CSV file not found at {csv_path}. Use `download=True`."
            )

        df = pd.read_csv(csv_path)
        labels = df["class"].values.astype(np.int64)
        images = df.drop(columns=["class"]).values.astype(np.float32)
        images = images.reshape(-1, 1, 28, 28)

        train_size = int(math.floor(len(images) * (1 - self.test_size)))
        if split == "train":
            images, labels = images[:train_size], labels[:train_size]
        else:
            images, labels = images[train_size:], labels[train_size:]

        if self.to_tensor:
            images = lucid.to_tensor(images, dtype=np.float32)
            labels = lucid.to_tensor(labels, dtype=np.int64)

        return images, labels

    def __getitem__(self, index: SupportsIndex) -> Tuple[Tensor, Tensor]:
        image = self.data[index].reshape(-1, 1, 28, 28)
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.data)
