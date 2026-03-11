from dataclasses import dataclass
from typing import Type
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor
from lucid.models.base import PreTrainedModelMixin

__all__ = ["LeNet", "LeNetConfig", "lenet_1", "lenet_4", "lenet_5"]


@dataclass
class LeNetConfig:
    conv_layers: list[dict[str, int]]
    clf_layers: list[int]
    clf_in_features: int
    base_activation: Type[nn.Module] = nn.Tanh

    def __post_init__(self) -> None:
        if len(self.conv_layers) != 2:
            raise ValueError("LeNet requires exactly 2 convolution layer configs.")

        for idx, layer_config in enumerate(self.conv_layers, start=1):
            if "out_channels" not in layer_config:
                raise ValueError(
                    f"LeNet conv layer config at index {idx} must define "
                    "'out_channels'."
                )

        if not self.clf_layers:
            raise ValueError("LeNet requires at least 1 classifier layer.")

        if self.clf_in_features <= 0:
            raise ValueError("LeNet clf_in_features must be greater than 0.")


class LeNet(nn.Module, PreTrainedModelMixin):
    def __init__(self, config: LeNetConfig) -> None:
        super().__init__()
        self.config = config

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, config.conv_layers[0]["out_channels"], kernel_size=5),
            config.base_activation(),
            nn.AvgPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                config.conv_layers[0]["out_channels"],
                config.conv_layers[1]["out_channels"],
                kernel_size=5,
            ),
            config.base_activation(),
            nn.AvgPool2d(2, 2),
        )

        in_features = config.clf_in_features
        n_clf_layers = len(config.clf_layers)
        for idx, units in enumerate(config.clf_layers, start=1):
            self.add_module(f"fc{idx}", nn.Linear(in_features, units))
            if idx < n_clf_layers:
                self.add_module(f"tanh{idx + 2}", config.base_activation())
            in_features = units

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)

        idx = 1
        while hasattr(self, f"fc{idx}"):
            x = getattr(self, f"fc{idx}")(x)
            if hasattr(self, f"tanh{idx + 2}"):
                x = getattr(self, f"tanh{idx + 2}")(x)
            idx += 1

        return x


def _lenet_config_from_kwargs(
    *,
    conv_layers: list[dict[str, int]],
    clf_layers: list[int],
    clf_in_features: int,
    **kwargs,
) -> LeNetConfig:
    legacy_base_activation = kwargs.pop("_base_activation", None)
    if legacy_base_activation is not None:
        if "base_activation" in kwargs:
            raise TypeError(
                "lenet factory received both '_base_activation' and "
                "'base_activation'."
            )
        kwargs["base_activation"] = legacy_base_activation

    return LeNetConfig(
        conv_layers=conv_layers,
        clf_layers=clf_layers,
        clf_in_features=clf_in_features,
        **kwargs,
    )


@register_model
def lenet_1(**kwargs) -> LeNet:
    config = _lenet_config_from_kwargs(
        conv_layers=[{"out_channels": 4}, {"out_channels": 12}],
        clf_layers=[10],
        clf_in_features=12 * 4 * 4,
        **kwargs,
    )
    return LeNet(config)


@register_model
def lenet_4(**kwargs) -> LeNet:
    config = _lenet_config_from_kwargs(
        conv_layers=[{"out_channels": 4}, {"out_channels": 12}],
        clf_layers=[84, 10],
        clf_in_features=12 * 4 * 4,
        **kwargs,
    )
    return LeNet(config)


@register_model
def lenet_5(**kwargs) -> LeNet:
    config = _lenet_config_from_kwargs(
        conv_layers=[{"out_channels": 6}, {"out_channels": 16}],
        clf_layers=[120, 84, 10],
        clf_in_features=16 * 5 * 5,
        **kwargs,
    )
    return LeNet(config)
