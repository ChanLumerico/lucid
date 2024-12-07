from typing import Any, Iterator, Self
from collections import OrderedDict
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
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if name in self._modules:
                del self._modules[name]
            self._parameters[name] = value

        elif isinstance(value, Module):
            if name in self._parameters:
                del self._parameters[name]
            self._modules[name] = value

        else:
            if name in self._parameters:
                del self._parameters[name]
            if name in self._modules:
                del self._modules[name]

        super().__setattr__(name, value)

    def add_module(self, name: str, module: Self) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module.")

        self.__setattr__(name, module)

    def register_parameter(self, name: str, param: Parameter) -> None:
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"{param} is not a Parameter.")

        self.__setattr__(name, param)

    def forward(self) -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError(
            "The forward method must be implemented by the subclass."
        )

    def parameters(self, recurse: bool = True) -> Iterator:
        for _, param in self._parameters.items():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=recurse)

    def modules(self) -> Iterator:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def state_dict(
        self,
        destination: OrderedDict | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Parameter]:
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            destination[prefix + name] = param

        for name, module in self._modules.items():
            module.state_dict(
                destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars
            )
        return destination

    def load_state_dict(
        self, state_dict: dict[str, Parameter], strict: bool = True
    ) -> None:
        own_state = self.state_dict()

        if strict and set(own_state.keys()) != set(state_dict.keys()):
            missing = set(own_state.keys()) - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - set(own_state.keys())
            msg = ""

            if missing:
                msg += f"Missing keys in state_dict: {missing}\n"
            if unexpected:
                msg += f"Unexpected keys in state_dict: {unexpected}\n"
            if msg:
                raise KeyError("Error(s) in loading state_dict:\n" + msg)

        for key, param in state_dict.items():
            splits = key.split(".")
            obj = self
            for split in splits[:-1]:
                obj = obj._modules[split]

            final_name = splits[-1]
            setattr(obj, final_name, param)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
        return self.forward(*args, **kwargs)
