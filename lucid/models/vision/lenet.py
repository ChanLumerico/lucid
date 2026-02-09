from typing import Type
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["LeNet", "lenet_1", "lenet_4", "lenet_5"]


class LeNet(nn.Module):
    def __init__(
        self,
        conv_layers: list[dict],
        clf_layers: list[int],
        clf_in_features: int,
        _base_activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_layers[0]["out_channels"], kernel_size=5),
            _base_activation(),
            nn.AvgPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                conv_layers[0]["out_channels"],
                conv_layers[1]["out_channels"],
                kernel_size=5,
            ),
            _base_activation(),
            nn.AvgPool2d(2, 2),
        )

        in_features = clf_in_features
        for idx, units in enumerate(clf_layers, start=1):
            self.add_module(f"fc{idx}", nn.Linear(in_features, units))
            self.add_module(f"tanh{idx + 2}", _base_activation())
            in_features = units

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)

        idx = 1
        while hasattr(self, f"fc{idx}"):
            x = getattr(self, f"fc{idx}")(x)
            x = getattr(self, f"tanh{idx + 2}")(x)
            idx += 1

        return x


@register_model
def lenet_1(**kwargs) -> LeNet:
    return LeNet(
        conv_layers=[{"out_channels": 4}, {"out_channels": 12}],
        clf_layers=[10],
        clf_in_features=12 * 4 * 4,
        **kwargs,
    )


@register_model
def lenet_4(**kwargs) -> LeNet:
    return LeNet(
        conv_layers=[{"out_channels": 4}, {"out_channels": 12}],
        clf_layers=[84, 10],
        clf_in_features=12 * 4 * 4,
        **kwargs,
    )


@register_model
def lenet_5(**kwargs) -> LeNet:
    return LeNet(
        conv_layers=[{"out_channels": 6}, {"out_channels": 16}],
        clf_layers=[120, 84, 10],
        clf_in_features=16 * 5 * 5,
        **kwargs,
    )
