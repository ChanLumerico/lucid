from typing import Callable
from functools import wraps

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor import Tensor


__all__ = [
    "Normalize",
    "Resize",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomCrop",
    "CenterCrop",
    "RandomRotation",
    "RandomGrayscale",
]


def add_batch_dim(func: Callable[..., Tensor]) -> Callable:
    @wraps(func)
    def wrapper(self, img: Tensor) -> Tensor:
        if img.ndim == 3:
            img = lucid.expand_dims(img, axis=0)
            ret = func(self, img)
            return ret.squeeze(axis=0)
        else:
            return func(self, img)

    return wrapper


class Normalize(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.mean = lucid.tensor(mean)
        self.std = lucid.tensor(std)

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        return (img - self.mean) / self.std


class Resize(nn.Module):
    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        return F.interpolate(img, size=self.size, mode="bilinear")


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        if lucid.random.uniform().item() < self.p:
            return img[:, :, :, ::-1]
        return img


class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        if lucid.random.uniform().item() < self.p:
            return img[:, :, ::-1, :]
        return img


class RandomCrop(nn.Module):
    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        _, _, H, W = img.shape
        crop_h, crop_w = self.size
        top = lucid.random.randint(0, H - crop_h + 1)
        left = lucid.random.randint(0, W - crop_w + 1)

        top = top.astype(int).squeeze().item()
        left = left.astype(int).squeeze().item()

        return img[:, :, top : top + crop_h, left : left + crop_w]


class CenterCrop(nn.Module):
    def __init__(self, size: int | tuple[int, int]) -> None:
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        _, _, H, W = img.shape
        crop_h, crop_w = self.size
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2

        return img[:, :, top : top + crop_h, left : left + crop_w]


class RandomRotation(nn.Module):
    def __init__(self, degrees: float) -> None:
        super().__init__()
        self.degrees = degrees

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        angle = lucid.random.uniform(-self.degrees, self.degrees)
        return lucid.nn.functional.rotate(img, angle.squeeze().item())


class RandomGrayscale(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    @add_batch_dim
    def forward(self, img: Tensor) -> Tensor:
        if lucid.random.uniform() < self.p:
            r, g, b = img[:, 0:1, ...], img[:, 1:2, ...], img[:, 2:3, ...]
            grayscale = 0.299 * r + 0.587 * g + 0.114 * b
            img = grayscale.repeat(3, axis=1)
        return img
