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

__all__ = [
    "LeNet_1_Weights",
    "LeNet_4_Weights",
    "LeNet_5_Weights",
    "AlexNet_Weights",
    "VGGNet_11_Weights",
    "VGGNet_13_Weights",
]
