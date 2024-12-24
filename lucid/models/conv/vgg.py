import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["VGGNet", "vggnet_11", "vggnet_13", "vggnet_16", "vggnet_19"]


class VGGNet(nn.Module):
    def __init__(self, conv_config: list[int | str], num_classes: int = 1000) -> None:
        super().__init__()
        self.conv = self._make_layers(conv_config)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, config: list[int | str]) -> nn.Sequential:
        layers = []
        in_channels = 3
        for layer in config:
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


@register_model
def vggnet_11(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, "M"]
    config.extend([128, "M"])
    config.extend([256, 256, "M"])
    config.extend([512, 512, "M", 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)


@register_model
def vggnet_13(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, "M"])
    config.extend([512, 512, "M", 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)


@register_model
def vggnet_16(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, 256, "M"])
    config.extend([512, 512, 512, "M", 512, 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)


@register_model
def vggnet_19(num_classes: int = 1000, **kwargs) -> VGGNet:
    config = [64, 64, "M"]
    config.extend([128, 128, "M"])
    config.extend([256, 256, 256, 256, "M"])
    config.extend([512, 512, 512, 512, "M", 512, 512, 512, 512, "M"])

    return VGGNet(config, num_classes, **kwargs)
