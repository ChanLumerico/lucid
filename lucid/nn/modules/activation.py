import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = [
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "ELU",
    "SELU",
    "GELU",
    "Sigmoid",
    "HardSigmoid",
    "Tanh",
    "Softmax",
    "Swish",
    "HardSwish",
]


class ReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return F.relu(input_)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size


class ReLU6(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return F.relu(lucid.clip(input_, 0, 6))

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 2


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input_: Tensor) -> Tensor:
        return F.leaky_relu(input_, self.negative_slope)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size


class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, input_: Tensor) -> Tensor:
        return F.elu(input_, self.alpha)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 2


class SELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return F.selu(input_)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 2


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return F.gelu(input_)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 8


class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return F.sigmoid(input_)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 4


class HardSigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return ((input_ / 6.0) + 0.5).clip(0, 1)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 3


class Tanh(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return F.tanh(input_)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 4


class Softmax(nn.Module):
    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, input_: Tensor) -> Tensor:
        return F.softmax(input_, self.axis)

    def __flops__(self, input_: Tensor) -> int | None:
        size = (
            input_.shape[self.axis]
            if self.axis >= 0
            else input_.shape[input_.ndim + self.axis]
        )
        return input_.size * 5 + size


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        return input_ * F.sigmoid(input_)

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 5


class HardSwish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: Tensor) -> Tensor:
        hard_sigmoid = ((input_ / 6.0) + 0.5).clip(0, 1)
        return input_ * hard_sigmoid

    def __flops__(self, input_: Tensor) -> int | None:
        return input_.size * 4
