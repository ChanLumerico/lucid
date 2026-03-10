from dataclasses import dataclass

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor
from lucid.models.base import PreTrainedModelMixin

__all__ = ["VGGNet", "VGGNetConfig", "vggnet_11", "vggnet_13", "vggnet_16", "vggnet_19"]


@dataclass
class VGGNetConfig:
    conv_config: list[int | str]
    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
    classifier_hidden_features: tuple[int, int] = (4096, 4096)

    def __post_init__(self) -> None:
        if not self.conv_config:
            raise ValueError("VGGNet conv_config must not be empty.")

        if self.num_classes <= 0:
            raise ValueError("VGGNet num_classes must be greater than 0.")

        if self.in_channels <= 0:
            raise ValueError("VGGNet in_channels must be greater than 0.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("VGGNet dropout must be in the range [0.0, 1.0).")

        if len(self.classifier_hidden_features) != 2:
            raise ValueError(
                "VGGNet classifier_hidden_features must contain exactly 2 values."
            )

        if any(hidden_dim <= 0 for hidden_dim in self.classifier_hidden_features):
            raise ValueError(
                "VGGNet classifier_hidden_features values must be greater than 0."
            )

        saw_conv = False
        for idx, layer in enumerate(self.conv_config, start=1):
            if layer == "M":
                continue
            if not isinstance(layer, int) or layer <= 0:
                raise ValueError(
                    "VGGNet conv_config entries must be positive integers or 'M' "
                    f"(got {layer!r} at index {idx})."
                )
            saw_conv = True

        if not saw_conv:
            raise ValueError("VGGNet conv_config must include at least 1 conv layer.")


class VGGNet(nn.Module, PreTrainedModelMixin):
    def __init__(self, config: VGGNetConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = self._make_layers(config)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        hidden_1, hidden_2 = config.classifier_hidden_features
        out_channels = next(
            layer
            for layer in reversed(config.conv_config)
            if isinstance(layer, int)
        )
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 7 * 7, hidden_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_2, config.num_classes),
        )

    def _make_layers(self, config: VGGNetConfig) -> nn.Sequential:
        layers = []
        in_channels = config.in_channels
        for layer in config.conv_config:
            if layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
                layers.append(nn.ReLU())

                in_channels = layer

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def _vgg_config_from_kwargs(
    *,
    conv_config: list[int | str],
    num_classes: int,
    **kwargs,
) -> VGGNetConfig:
    return VGGNetConfig(conv_config=conv_config, num_classes=num_classes, **kwargs)


@register_model
def vggnet_11(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, "M"]
    config.extend([128, "M"])
    config.extend([256, 256, "M"])
    config.extend([512, 512, "M", 512, 512, "M"])

    model_config = _vgg_config_from_kwargs(
        conv_config=config,
        num_classes=num_classes,
        **kwargs,
    )
    return VGGNet(model_config)


@register_model
def vggnet_13(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, "M"])
    config.extend([512, 512, "M", 512, 512, "M"])

    model_config = _vgg_config_from_kwargs(
        conv_config=config,
        num_classes=num_classes,
        **kwargs,
    )
    return VGGNet(model_config)


@register_model
def vggnet_16(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, 256, "M"])
    config.extend([512, 512, 512, "M", 512, 512, 512, "M"])

    model_config = _vgg_config_from_kwargs(
        conv_config=config,
        num_classes=num_classes,
        **kwargs,
    )
    return VGGNet(model_config)


@register_model
def vggnet_19(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, 256, 256, "M"])
    config.extend([512, 512, 512, 512, "M", 512, 512, 512, 512, "M"])

    model_config = _vgg_config_from_kwargs(
        conv_config=config,
        num_classes=num_classes,
        **kwargs,
    )
    return VGGNet(model_config)
