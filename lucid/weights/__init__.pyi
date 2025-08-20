from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class WeightEntry:
    url: str
    sha256: str
    tag: str
    dataset: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class LeNet_1_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

class LeNet_4_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

class LeNet_5_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

class AlexNet_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_11_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_13_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_16_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class VGGNet_19_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_18_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_34_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_50_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_101_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNet_152_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class Wide_ResNet_50_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class Wide_ResNet_101_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNeXt_50_32X4D_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNeXt_101_32X8D_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class ResNeXt_101_64X4D_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class DenseNet_121_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class DenseNet_169_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class DenseNet_201_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class MobileNet_V2_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class MobileNet_V3_Small_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

class MobileNet_V3_Large_Weights(Enum):
    IMAGENET1K: WeightEntry
    DEFAULT: WeightEntry

__all__ = [
    "LeNet_1_Weights",
    "LeNet_4_Weights",
    "LeNet_5_Weights",
    "AlexNet_Weights",
    "VGGNet_11_Weights",
    "VGGNet_13_Weights",
    "VGGNet_16_Weights",
    "VGGNet_19_Weights",
    "ResNet_18_Weights",
    "ResNet_34_Weights",
    "ResNet_50_Weights",
    "ResNet_101_Weights",
    "ResNet_152_Weights",
    "Wide_ResNet_50_Weights",
    "Wide_ResNet_101_Weights",
    "ResNeXt_50_32X4D_Weights",
    "ResNeXt_101_32X8D_Weights",
    "ResNeXt_101_64X4D_Weights",
    "DenseNet_121_Weights",
    "DenseNet_169_Weights",
    "DenseNet_201_Weights",
    "MobileNet_V2_Weights",
    "MobileNet_V3_Small_Weights",
    "MobileNet_V3_Large_Weights",
]
