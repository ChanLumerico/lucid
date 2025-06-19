import lucid.nn as nn

from lucid import register_model
from .resnet import ResNet, _Bottleneck


__all__ = [
    "ResNeXt",
    "resnext_50_32x4d",
    "resnext_101_32x4d",
    "resnext_101_32x8d",
    "resnext_101_32x16d",
    "resnext_101_32x32d",
    "resnext_101_64x4d",
]


class ResNeXt(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        cardinality: int,
        base_width: int,
        num_classes: int = 1000,
    ) -> None:
        block_args = {"cardinality": cardinality, "base_width": base_width}
        super().__init__(block, layers, num_classes, block_args=block_args)


@register_model
def resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 6, 3]
    return ResNeXt(_Bottleneck, layers, 32, 4, num_classes, **kwargs)


@register_model
def resnext_101_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 4, num_classes, **kwargs)


@register_model
def resnext_101_32x8d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 8, num_classes, **kwargs)


@register_model
def resnext_101_32x16d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 16, num_classes, **kwargs)


@register_model
def resnext_101_32x32d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 32, 32, num_classes, **kwargs)


@register_model
def resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> ResNeXt:
    layers = [3, 4, 23, 3]
    return ResNeXt(_Bottleneck, layers, 64, 4, num_classes, **kwargs)
