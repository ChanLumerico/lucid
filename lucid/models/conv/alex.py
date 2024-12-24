import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["AlexNet", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


@register_model
def alexnet(num_classes: int = 1000, **kwargs) -> AlexNet:
    return AlexNet(num_classes, **kwargs)
