from typing import Any, Iterator, Self, Type
from collections import OrderedDict
import numpy as np

from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar


class Parameter(Tensor):
    def __init__(self, data: Tensor | _ArrayOrScalar, dtype=np.float32) -> None:
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=True, keep_grad=True, dtype=dtype)


class Buffer(Tensor):
    def __init__(self, data: Tensor | _ArrayOrScalar, dtype=np.float32) -> None:
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=False, keep_grad=False, dtype=dtype)


class Module:
    _registry_map: dict[Type, OrderedDict[str, Any]] = {}

    def __init__(self) -> None:
        self._parameters: OrderedDict[str, Parameter]
        self._buffers: OrderedDict[str, Buffer]
        self._modules: OrderedDict[str, Self]

        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

        self.training = True

    def __setattr__(self, name: str, value: Any) -> None:
        registry_map: dict[Type, OrderedDict[str, Any]] = {
            Parameter: self._parameters,
            Buffer: self._buffers,
            Module: self._modules,
        }

        target_registry = None
        for cls, registry in registry_map.items():
            if isinstance(value, cls):
                target_registry = registry
                break

        if target_registry is not None:
            for registry in registry_map.values():
                if registry is not target_registry and name in registry:
                    del registry[name]
            target_registry[name] = value
        else:
            for registry in registry_map.values():
                if name in registry:
                    del registry[name]

        super().__setattr__(name, value)

    def add_module(self, name: str, module: Self) -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module.")

        self.__setattr__(name, module)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"{param} is not a Parameter.")

        self.__setattr__(name, param)

    def register_buffer(
        self, name: str, buffer: Buffer | _ArrayOrScalar | None, dtype: Any = np.float32
    ) -> None:
        if buffer is not None:
            if not isinstance(buffer, Buffer):
                buffer = Buffer(buffer, dtype=dtype)

        self.__setattr__(name, buffer)

    def forward(self) -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError(
            "The forward method must be implemented by the subclass."
        )

    def train(self, mode: bool = True) -> Self:
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> Self:
        return self.train(mode=False)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self._parameters.items():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[Buffer]:
        for buffer in self._buffers.values():
            yield buffer
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse=recurse)

    def modules(self) -> Iterator[Self]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def state_dict(
        self,
        destination: OrderedDict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            destination[prefix + name] = param if keep_vars else param.data

        for name, buffer in self._buffers.items():
            destination[prefix + name] = buffer if keep_vars else buffer.data

        for name, module in self._modules.items():
            module.state_dict(
                destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars
            )

        return destination

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        own_state = self.state_dict(keep_vars=True)

        missing_keys = set(own_state.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(own_state.keys())

        if strict:
            msg = ""
            if missing_keys:
                msg += f"Missing keys in state_dict: {missing_keys}\n"
            if unexpected_keys:
                msg += f"Unexpected keys in state_dict: {unexpected_keys}\n"
            if msg:
                raise KeyError("Error(s) in loading state_dict:\n" + msg)

        for key, value in state_dict.items():
            if key in own_state:
                attr = own_state[key]
                if isinstance(attr, (Parameter, Buffer)):
                    if isinstance(value, Tensor):
                        attr.data = value.data
                    else:
                        attr.data = value
                else:
                    setattr(self, key, value)
            elif strict:
                raise KeyError(f"Unexpected key '{key}' in state_dict.")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
        return self.forward(*args, **kwargs)
