from typing import (
    Any,
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Self,
    Type,
    ValuesView,
    overload,
)
from collections import OrderedDict

from lucid._tensor import Tensor
from lucid.types import _ArrayOrScalar, _StateDict, _NumPyArray, _DeviceType

import lucid
import lucid.nn as nn


__all__ = [
    "Module",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
]

_ForwardHookType = Callable[["Module", tuple[Tensor], tuple[Tensor]], None]
_BackwardHookType = Callable[[Tensor, _NumPyArray], None]


class Module:
    _registry_map: dict[Type, OrderedDict[str, Any]] = {}
    _alt_name: str = ""

    def __init__(self) -> None:
        self._parameters: OrderedDict[str, nn.Parameter]
        self._buffers: OrderedDict[str, nn.Buffer]
        self._modules: OrderedDict[str, Self]

        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

        self.training = True
        self.device: _DeviceType = "cpu"

        self._forward_hooks: list[_ForwardHookType] = []
        self._backward_hooks: list[_BackwardHookType] = []

    def __setattr__(self, name: str, value: Any) -> None:
        registry_map: dict[Type, OrderedDict[str, Any]] = {
            nn.Parameter: self._parameters,
            nn.Buffer: self._buffers,
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

    def register_parameter(self, name: str, param: nn.Parameter | None) -> None:
        if not isinstance(param, nn.Parameter) and param is not None:
            raise TypeError(f"{param} is not a nn.Parameter.")

        self.__setattr__(name, param)

    def register_buffer(
        self,
        name: str,
        buffer: nn.Buffer | _ArrayOrScalar | None,
        dtype: type | None = None,
    ) -> None:
        if buffer is not None:
            if not isinstance(buffer, nn.Buffer):
                buffer = nn.Buffer(buffer, dtype=dtype, device=self.device)

        self.__setattr__(name, buffer)

    def register_forward_hook(self, hook: _ForwardHookType) -> Callable:
        self._forward_hooks.append(hook)
        return lambda: self._forward_hooks.remove(hook)

    def register_backward_hook(self, hook: _BackwardHookType) -> Callable:
        self._backward_hooks.append(hook)
        return lambda: self._backward_hooks.remove(hook)

    def reset_parameters(self) -> None:
        for param in self.parameters():
            param.zero()

    def forward(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
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

    def to(self, device: _DeviceType) -> Self:
        if device == self.device:
            return self
        self.device = device

        for param in self.parameters(recurse=False):
            param.to(device)

        for buffer in self.buffers(recurse=False):
            buffer.to(device)

        for module in self.modules():
            module.to(device)

        return self

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for _, param in self._parameters.items():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[nn.Buffer]:
        for buffer in self._buffers.values():
            yield buffer
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse=recurse)

    def modules(self) -> Iterator[Self]:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def count_parameters(self, recurse: bool = True) -> int:
        total_params = sum(p.size for p in self.parameters(recurse=recurse))
        return total_params

    @property
    def parameter_size(self) -> int:
        return self.count_parameters(recurse=True)

    def apply(self, fn: Callable[[Self, Any], None]) -> Self:
        fn(self)
        for module in self._modules.values():
            module.apply(fn)
        return self

    def state_dict(
        self,
        destination: OrderedDict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> _StateDict:
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

    def load_state_dict(self, state_dict: _StateDict, strict: bool = True) -> None:
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
                if isinstance(attr, (nn.Parameter, nn.Buffer)):
                    if isinstance(value, Tensor):
                        attr.data = value.data
                    else:
                        attr.data = value
                else:
                    setattr(self, key, value)
            elif strict:
                raise KeyError(f"Unexpected key '{key}' in state_dict.")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
        output = self.forward(*args, **kwargs)
        for hook in self._forward_hooks:
            hook(self, args, output)

        if isinstance(output, Tensor) and self._backward_hooks:
            for hook in self._backward_hooks:
                output.register_hook(hook)

        return output


class Sequential(Module):
    @overload
    def __init__(self, *modules: Module) -> None: ...

    @overload
    def __init__(self, ordered_dict: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args: Module | OrderedDict[str, Module]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.add_module(name, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input: Tensor) -> Tensor:
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx: int | slice) -> Module | Self:
        if isinstance(idx, slice):
            modules_slice = list(self._modules.items())[idx]
            return Sequential(OrderedDict(modules_slice))

        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self._modules)
            keys = list(self._modules.keys())

            if idx < 0 or idx >= len(keys):
                raise IndexError("Index out of range")

            return self._modules[keys[idx]]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Must be int or slice.")

    def __setitem__(self, idx: int, module: Module) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers for __setitem__.")

        keys = list(self._modules.keys())
        if idx < 0:
            idx += len(keys)
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range")

        old_key = keys[idx]
        del self._modules[old_key]
        self._modules[old_key] = module

    def __delitem__(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers for __delitem__.")

        keys = list(self._modules.keys())
        if idx < 0:
            idx += len(keys)
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range")

        del self._modules[keys[idx]]

    def __len__(self) -> int:
        return len(self._modules)

    def append(self, module: Module) -> None:
        self.add_module(str(len(self._modules)), module)

    def extend(self, modules: Iterator[Module]) -> None:
        for module in modules:
            self.append(module)

    @classmethod
    def from_ordered_dict(cls: type[Self], odict: OrderedDict[str, Module]) -> Self:
        return cls(odict)

    @classmethod
    def from_modules(cls: type[Self], *modules: Module) -> Self:
        return cls(*modules)


class ModuleList(Module):
    def __init__(self, modules: list[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx: int | slice) -> Module | Self:
        if isinstance(idx, slice):
            mod_slice = list(self._modules.items())[idx]
            ml = ModuleList()
            for i, (_, m) in enumerate(mod_slice):
                ml.add_module(str(i), m)
            return ml

        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self._modules)
            keys = list(self._modules.keys())
            if idx < 0 or idx >= len(keys):
                raise IndexError("Index out of range.")

            return self._modules[keys[idx]]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}. Must be int or slice.")

    def __setitem__(self, idx: int, module: Module) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers.")

        keys = list(self._modules.keys())
        if idx < 0:
            idx += len(keys)
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range.")

        old_key = keys[idx]
        del self._modules[old_key]
        self.add_module(old_key, module)

    def __delitem__(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers.")

        keys = list(self._modules.keys())
        if idx < 0:
            idx += len(keys)
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range.")

        items = list(self._modules.items())
        self._modules.clear()
        for i, (_, m) in enumerate(items):
            self._modules[str(i)] = m

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> None:
        self.add_module(str(len(self._modules)), module)

    def extend(self, modules: list[Module]) -> None:
        for m in modules:
            self.append(m)

    def insert(self, index: int, module: Module) -> None:
        if not isinstance(index, int):
            raise TypeError("Index should be an integer.")

        total = len(self._modules)
        if index < 0:
            index += total
        if index < 0:
            index = 0
        if index > total:
            index = total

        items = list(self._modules.items())
        items.insert(index, (str(index), module))

        self._modules.clear()
        for i, (_, m) in enumerate(items):
            self._modules[str(i)] = m


class ModuleDict(Module):
    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def update(self, modules: dict[str, Module]) -> None:
        for k, m in modules.items():
            self[k] = m

    def clear(self) -> None:
        self._modules.clear()

    def pop(self, key: str) -> Module:
        module = self._modules[key]
        del self._modules[key]
        return module

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"Expected Module, got {type(module)}.")
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def keys(self) -> KeysView[str]:
        return self._modules.keys()

    def values(self) -> ValuesView[Module]:
        return self._modules.values()

    def items(self) -> ItemsView[str, Module]:
        return self._modules.items()


class ParameterList(Module):
    def __init__(self, parameters: list[nn.Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def __getitem__(self, idx: int | slice) -> nn.Parameter | Self:
        if isinstance(idx, slice):
            items = list(self._parameters.items())[idx]
            plist = ParameterList()

            for i, (_, p) in enumerate(items):
                plist.register_parameter(str(i), p)
            return plist

        elif isinstance(idx, int):
            if idx < 0:
                idx += len(self._parameters)
            keys = list(self._parameters.keys())

            if idx < 0 or idx >= len(keys):
                raise IndexError("Index out of range.")
            return self._parameters[keys[idx]]

        else:
            return TypeError(f"Invalid index type: {type(idx)}. Must be int or slice.")

    def __setitem__(self, idx: int, param: nn.Parameter) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers.")
        if not isinstance(param, nn.Parameter):
            raise TypeError("Can only set Parameter in ParameterList.")

        keys = list(self._parameters.keys())
        if idx < 0:
            idx += len(keys)
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range.")

        old_key = keys[idx]
        del self._parameters[old_key]
        self.register_parameter(old_key, param)

    def __delitem__(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers.")

        keys = list(self._parameters.keys())
        if idx < 0:
            idx += len(keys)
        if idx < 0 or idx >= len(keys):
            raise IndexError("Index out of range")

        del self._parameters[keys[idx]]

        items = list(self._parameters.items())
        self._parameters.clear()
        for i, (_, p) in enumerate(items):
            self._parameters[str(i)] = p

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters.values())

    def append(self, param: nn.Parameter) -> None:
        if not isinstance(param, nn.Parameter):
            raise TypeError("Can only append Parameter to ParameterList.")
        self.register_parameter(str(len(self._parameters)), param)

    def extend(self, parameters) -> None:
        for p in parameters:
            self.append(p)

    def insert(self, index: int, param: nn.Parameter) -> None:
        if not isinstance(index, int):
            raise TypeError("Index should be an integer for insert.")
        if not isinstance(param, nn.Parameter):
            raise TypeError("Can only insert Parameter into ParameterList.")

        total = len(self._parameters)
        if index < 0:
            index += total
        if index < 0:
            index = 0
        if index > total:
            index = total

        items = list(self._parameters.items())
        items.insert(index, (str(index), param))

        self._parameters.clear()
        for i, (_, p) in enumerate(items):
            self._parameters[str(i)] = p


class ParameterDict(Module):
    def __init__(self, parameters: dict[str, nn.Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            self.update(parameters)

    def update(self, parameters: dict[str, nn.Parameter]) -> None:
        for k, p in parameters.items():
            self[k] = p

    def clear(self) -> None:
        self._parameters.clear()

    def pop(self, key: str) -> nn.Parameter:
        param = self._parameters[key]
        del self._parameters[key]
        return param

    def __getitem__(self, key: str) -> nn.Parameter:
        return self._parameters[key]

    def __setitem__(self, key: str, param: nn.Parameter) -> None:
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"Expected nn.Parameter, got {type(param)}")
        self.register_parameter(key, param)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def keys(self) -> KeysView[str]:
        return self._parameters.keys()

    def values(self) -> ValuesView[nn.Parameter]:
        return self._parameters.values()

    def items(self) -> ItemsView[str, nn.Parameter]:
        return self._parameters.items()
