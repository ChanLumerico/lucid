from abc import abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union, ClassVar

from lucid.transforms import Compose
from lucid.data import Dataset
from lucid._tensor import Tensor

import lucid.nn as nn


class DatasetBase(Dataset):
    OPENML_ID: ClassVar[int]

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

    @abstractmethod
    def _download(self) -> None: ...

    @abstractmethod
    def _load_data(self, *args, **kwargs) -> Tuple[Tensor, ...]: ...

    def __len__(self) -> int:
        return len(self.data)
