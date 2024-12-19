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
    "RandomGrayscale",
]


class Normalize(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.mean = lucid.tensor(mean)
        self.std = lucid.tensor(std)

    def forward(self, img: Tensor) -> Tensor:
        return (img - self.mean) / self.std


class Resize(nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: Tensor) -> Tensor:
        return F.interpolate(img, size=self.size, mode="bilinear")


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        if lucid.random.uniform().item() < self.p:
            return img[:, :, :, ::-1]
        return img


class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        if lucid.random.uniform().item() < self.p:
            return img[:, :, ::-1, :]
        return img


class RandomCrop(nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: Tensor) -> Tensor:
        _, _, H, W = img.shape
        crop_h, crop_w = self.size
        top = lucid.random.randint(0, H - crop_h + 1)
        left = lucid.random.randint(0, W - crop_w + 1)

        return img[:, :, top : top + crop_h, left : left + crop_w]


class CenterCrop(nn.Module):
    def __init__(self, size: tuple[int, int]) -> None:
        super().__init__()
        self.size = size

    def forward(self, img: Tensor) -> Tensor:
        _, _, H, W = img.shape
        crop_h, crop_w = self.size
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2

        return img[:, :, top : top + crop_h, left : left + crop_w]


class RandomGrayscale(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, img: Tensor) -> Tensor:
        if lucid.random.uniform() < self.p:
            img = img.mean(axis=1, keepdims=True)
            img = img.repeat(3, axis=1)
        return img


class ColorJitter(nn.Module):
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, img: Tensor) -> Tensor:
        if self.brightness > 0:
            img *= 1 + lucid.random.uniform(-self.brightness, self.brightness)
        if self.contrast > 0:
            mean = img.mean()
            img = (img - mean) * (
                1 + lucid.random.uniform(-self.contrast, self.contrast)
            ) + mean
        if self.saturation > 0:
            img *= 1 + lucid.random.uniform(-self.saturation, self.saturation)
        if self.hue > 0:
            img += lucid.random.uniform(-self.hue, self.hue)
        return img
