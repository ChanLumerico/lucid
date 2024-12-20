import os
import pandas as pd
import numpy as np
from urllib import request
from pathlib import Path
from typing import Optional, Tuple, Union

import lucid
from lucid.data import Dataset
from lucid._tensor import Tensor


__all__ = ["MNIST"]


class MNIST(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: Optional[bool] = True,
        download: Optional[bool] = False,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        if self.train:
            self.data, self.targets = self._load_data("train")
        else:
            self.data, self.targets = self._load_data("test")

    def _download(self) -> None:
        urls = {...}  # TODO: Need to be fixed.

        self.root.mkdir(parents=True, exist_ok=True)

        for _, url in urls.items():
            file_name = url.split("/")[-1]
            file_path = self.root / file_name
            if not file_path.exists():
                print(f"Downloading {url} to {file_path}")
                try:
                    request.urlretrieve(url, file_path)
                    print(f"Successfully downloaded {file_path}")
                except Exception as e:
                    print(f"Failed to download {url}. Error: {e}")

    def _load_data(self, split: str) -> Tuple[Tensor, Tensor]:
        if split == "train":
            images_path = self.root / "MNIST_train.csv"
            labels_path = self.root / "MNIST_train_labels.csv"
        else:
            images_path = self.root / "MNIST_test.csv"
            labels_path = self.root / "MNIST_test_labels.csv"

        try:
            images_df = pd.read_csv(images_path, header=None)
            labels_df = pd.read_csv(labels_path, header=None)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MNIST {split} data. Ensure files exist. Error: {e}"
            )

        images = images_df.to_numpy().reshape(-1, 28, 28)
        labels = labels_df.to_numpy().flatten()

        images_t = lucid.to_tensor(images, dtype=np.float32)
        labels_t = lucid.to_tensor(labels, dtype=np.int64)

        return images_t, labels_t

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = self.data[index]
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.data)
