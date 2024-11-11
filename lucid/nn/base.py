from typing import Any, Self
import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar


class Parameter(Tensor):
    def __init__(self, data: Tensor | _ArrayOrScalar, dtype=np.float32):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=True, keep_grad=True, dtype=dtype)


class Module:
    def __init__(self) -> None:
        self._params: list[Parameter] = []
        self._modules: list[Module] = []

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            self._params.append((name, value))
        elif isinstance(value, Module):
            self._modules.append((name, value))

        super().__setattr__(name, value)

    def forward(self) -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError(
            "The forward method must be implemented by the subclass."
        )

    def parameters(self):
        return [param for _, param in self._params]

    def modules(self):
        return [module for _, module in self._modules]

    def state_dict(self) -> dict[str, Parameter | Self]:
        state_dict = {}
        for name, param in self._parameters:
            state_dict[name] = param
        for name, module in self._modules:
            state_dict[name] = module.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Parameter | Self]) -> None:
        for name, param in state_dict.items():
            setattr(self, name, param)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
        return self.forward(*args, **kwargs)
