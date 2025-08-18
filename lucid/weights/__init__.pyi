from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class WeightEntry:
    url: str
    sha256: str
    tag: str
    size: Optional[int] = None
    dataset: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class LeNet_5_Weights(Enum):
    MNIST: WeightEntry
    DEFAULT: WeightEntry

__all__ = [
    "LeNet_5_Weights",
]
