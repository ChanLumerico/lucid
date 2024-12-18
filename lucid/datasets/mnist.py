import os
import gzip
import numpy as np
from urllib import request
from pathlib import Path

import lucid
from lucid.data import Dataset
from lucid._tensor import Tensor


__all__ = ["MNIST"]


class MNIST(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = False,
        **__transform_kwargs,  # NOTE: Support this after impl. of `lucid.transforms`
    ) -> None:
        self.root = root
        self.train = train

        if download:
            self._download()

        if self.train:
            self.data, self.targets = self._load_data("train")
        else:
            self.data, self.targets = self._load_data("test")

    def _download(self) -> None:
        urls = {  # NOTE: It seems these links are forbidden. Try for other mirrors.
            "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        }

        os.makedirs(self.root, exist_ok=True)
        for _, url in urls.items():
            file_path = os.path.join(self.root, url.split("/")[-1])

            if not os.path.exists(file_path):
                print(f"Downloading {url} to {file_path}")
                request.urlretrieve(url, file_path)

    def _load_data(self, split: str) -> tuple[Tensor, Tensor]:
        if split == "train":
            images_path = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")

        with gzip.open(images_path, "rb") as img_path:
            images = np.frombuffer(img_path.read(), np.uint8, offset=16).reshape(
                -1, 28, 28
            )

        with gzip.open(labels_path, "rb") as lbl_path:
            labels = np.frombuffer(lbl_path.read(), np.uint8, offset=8)

        images_t = lucid.to_tensor(images, dtype=images.dtype)
        labels_t = lucid.to_tensor(labels, dtype=labels.dtype)

        return images_t, labels_t

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = self.data[index]
        label = self.targets[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.data)
