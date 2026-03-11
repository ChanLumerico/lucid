from dataclasses import dataclass, field
from typing import Any, Literal

from lucid import register_model
from .resnet import ResNet, ResNetConfig
from lucid.models.base import PreTrainedModelMixin

__all__ = [
    "ResNeXt",
    "ResNeXtConfig",
    "resnext_50_32x4d",
    "resnext_101_32x4d",
    "resnext_101_32x8d",
    "resnext_101_32x16d",
    "resnext_101_32x32d",
    "resnext_101_64x4d",
]


@dataclass
class ResNeXtConfig:
    layers: tuple[int, int, int, int] | list[int]
    cardinality: int
    base_width: int
    num_classes: int = 1000
    in_channels: int = 3
    stem_width: int = 64
    stem_type: Literal["deep"] | None = None
    avg_down: bool = False
    channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
    block_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        self.channels = tuple(self.channels)
        if not isinstance(self.block_args, dict):
            raise TypeError("block_args must be a dictionary")
        self.block_args = dict(self.block_args)

        if len(self.layers) != 4:
            raise ValueError("layers must contain exactly 4 stage depths")
        if any(not isinstance(depth, int) or depth <= 0 for depth in self.layers):
            raise ValueError("layers values must be positive integers")
        if self.cardinality <= 0:
            raise ValueError("cardinality must be greater than 0")
        if self.base_width <= 0:
            raise ValueError("base_width must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.stem_width <= 0:
            raise ValueError("stem_width must be greater than 0")
        if self.stem_type not in (None, "deep"):
            raise ValueError("stem_type must be None or 'deep'")
        if len(self.channels) != 4:
            raise ValueError("channels must contain exactly 4 stage widths")
        if any(
            not isinstance(channel, int) or channel <= 0 for channel in self.channels
        ):
            raise ValueError("channels values must be positive integers")


class ResNeXt(ResNet, PreTrainedModelMixin):
    def __init__(self, config: ResNeXtConfig) -> None:
        block_args = {
            "cardinality": config.cardinality,
            "base_width": config.base_width,
            **config.block_args,
        }
        super().__init__(
            ResNetConfig(
                block="bottleneck",
                layers=config.layers,
                num_classes=config.num_classes,
                in_channels=config.in_channels,
                stem_width=config.stem_width,
                stem_type=config.stem_type,
                avg_down=config.avg_down,
                channels=config.channels,
                block_args=block_args,
            )
        )
        self.config = config
        self.cardinality = config.cardinality
        self.base_width = config.base_width


def _build_resnext_config(
    *,
    layers: tuple[int, int, int, int] | list[int],
    cardinality: int,
    base_width: int,
    num_classes: int,
    kwargs: dict[str, Any] | None = None,
) -> ResNeXtConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {"layers", "cardinality", "base_width"}
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset layers, cardinality, or base_width"
        )

    return ResNeXtConfig(
        layers=layers,
        cardinality=cardinality,
        base_width=base_width,
        num_classes=num_classes,
        **kwargs,
    )


@register_model
def resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 6, 3]
    config = _build_resnext_config(
        layers=layers,
        cardinality=32,
        base_width=4,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeXt(config)


@register_model
def resnext_101_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    config = _build_resnext_config(
        layers=layers,
        cardinality=32,
        base_width=4,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeXt(config)


@register_model
def resnext_101_32x8d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    config = _build_resnext_config(
        layers=layers,
        cardinality=32,
        base_width=8,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeXt(config)


@register_model
def resnext_101_32x16d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    config = _build_resnext_config(
        layers=layers,
        cardinality=32,
        base_width=16,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeXt(config)


@register_model
def resnext_101_32x32d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    config = _build_resnext_config(
        layers=layers,
        cardinality=32,
        base_width=32,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeXt(config)


@register_model
def resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    config = _build_resnext_config(
        layers=layers,
        cardinality=64,
        base_width=4,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeXt(config)
