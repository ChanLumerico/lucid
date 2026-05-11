"""ZFNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ZFNetConfig(ModelConfig):
    """Configuration for ZFNet (Zeiler & Fergus, 2013).

    ZFNet is an AlexNet variant with modified first two conv layers:
    - Conv1: 7×7 stride=2 pad=1 (vs AlexNet's 11×11 stride=4)
    - Conv2: 5×5 stride=2 (vs AlexNet's 5×5 stride=1)
    """

    model_type: ClassVar[str] = "zfnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
