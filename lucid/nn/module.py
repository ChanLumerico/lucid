from typing import (
    Any,
    Callable,
    ItemsView,
    Iterator,
    KeysView,
    Self,
    Type,
    TypeVar,
    ValuesView,
    overload,
)
from collections import OrderedDict

from lucid._tensor import Tensor
from lucid.types import (
    _ArrayOrScalar,
    _BackwardHook,
    _DeviceType,
    _ForwardHook,
    _ForwardHookKwargs,
    _ForwardPreHook,
    _ForwardPreHookKwargs,
    _FullBackwardHook,
    _FullBackwardPreHook,
    _LoadStateDictPostHook,
    _LoadStateDictPreHook,
    _NumPyArray,
    _StateDictHook,
    _StateDictPreHook,
)

import lucid.nn as nn


__all__ = [
    "Module",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
    "auto_repr",
    "set_state_dict_pass_attr",
]


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

        self._forward_pre_hooks: list[
            tuple[_ForwardPreHook | _ForwardPreHookKwargs, bool]
        ] = []
        self._forward_hooks: list[tuple[_ForwardHook | _ForwardHookKwargs, bool]] = []

        self._backward_hooks: list[_BackwardHook] = []
        self._full_backward_pre_hooks: list[_FullBackwardPreHook] = []
        self._full_backward_hooks: list[_FullBackwardHook] = []

        self._state_dict_pre_hooks: list[_StateDictPreHook] = []
        self._state_dict_hooks: list[_StateDictHook] = []

        self._load_state_dict_pre_hooks: list[_LoadStateDictPreHook] = []
        self._load_state_dict_post_hooks: list[_LoadStateDictPostHook] = []

        self._state_dict_pass_attr = set()

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

    def setattr_raw(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

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

    def register_forward_pre_hook(
        self,
        hook: _ForwardPreHook | _ForwardPreHookKwargs,
        *,
        with_kwargs: bool = False,
    ) -> Callable:
        self._forward_pre_hooks.append((hook, with_kwargs))
        return lambda: self._forward_pre_hooks.remove((hook, with_kwargs))

    def register_forward_hook(
        self, hook: _ForwardHook | _ForwardHookKwargs, *, with_kwargs: bool = False
    ) -> Callable:
        self._forward_hooks.append((hook, with_kwargs))
        return lambda: self._forward_hooks.remove((hook, with_kwargs))

    def register_backward_hook(self, hook: _BackwardHook) -> Callable:
        self._backward_hooks.append(hook)
        return lambda: self._backward_hooks.remove(hook)

    def register_full_backward_pre_hook(self, hook: _FullBackwardPreHook) -> Callable:
        self._full_backward_pre_hooks.append(hook)
        return lambda: self._full_backward_pre_hooks.remove(hook)

    def register_full_backward_hook(self, hook: _FullBackwardHook) -> Callable:
        self._full_backward_hooks.append(hook)
        return lambda: self._full_backward_hooks.remove(hook)

    def register_state_dict_pre_hook(self, hook: _StateDictPreHook) -> Callable:
        self._state_dict_pre_hooks.append(hook)
        return lambda: self._state_dict_pre_hooks.remove(hook)

    def register_state_dict_hook(self, hook: _StateDictHook) -> Callable:
        self._state_dict_hooks.append(hook)
        return lambda: self._state_dict_hooks.remove(hook)

    def register_load_state_dict_pre_hook(
        self, hook: _LoadStateDictPreHook
    ) -> Callable:
        self._load_state_dict_pre_hooks.append(hook)
        return lambda: self._load_state_dict_pre_hooks.remove(hook)

    def register_load_state_dict_post_hook(
        self, hook: _LoadStateDictPostHook
    ) -> Callable:
        self._load_state_dict_post_hooks.append(hook)
        return lambda: self._load_state_dict_post_hooks.remove(hook)

    def reset_parameters(self) -> None:
        for param in self.parameters():
            param.zero()

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

    def children(self: Self) -> Iterator[Self]:
        return iter(self._modules.values())

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
    ) -> OrderedDict:
        for hook in self._state_dict_pre_hooks:
            hook(self, prefix, keep_vars)

        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            destination[prefix + name] = param if keep_vars else param.numpy()

        for name, buffer in self._buffers.items():
            destination[prefix + name] = buffer if keep_vars else buffer.numpy()

        for name, module in self._modules.items():
            module.state_dict(
                destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars
            )

        for key in list(destination.keys()):
            if key in self._state_dict_pass_attr:
                del destination[key]

        for hook in self._state_dict_hooks:
            hook(self, destination, prefix, keep_vars)

        return destination

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True) -> None:
        for hook in self._load_state_dict_pre_hooks:
            hook(self, state_dict, strict)

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
                    value_t = Tensor(value, device=self.device)
                    attr.data = value_t.data
                else:
                    setattr(self, key, value)
            elif strict:
                raise KeyError(f"Unexpected key '{key}' in state_dict.")

        for hook in self._load_state_dict_post_hooks:
            hook(self, missing_keys, unexpected_keys, strict)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor | tuple[Tensor, ...]:
        for hook, with_kwargs in self._forward_pre_hooks:
            if with_kwargs:
                result = hook(self, args, kwargs)
                if result is not None:
                    args, kwargs = result
            else:
                result = hook(self, args)
                if result is not None:
                    args = result

        output = self.forward(*args, **kwargs)

        for hook, with_kwargs in self._forward_hooks:
            if with_kwargs:
                result = hook(self, args, kwargs, output)
            else:
                result = hook(self, args, output)
            if result is not None:
                output = result

        if isinstance(output, Tensor) and self._backward_hooks:
            for hook in self._backward_hooks:
                output.register_hook(hook)

        if self._full_backward_pre_hooks or self._full_backward_hooks:
            outputs = output if isinstance(output, tuple) else (output,)
            output_tensors = [out for out in outputs if isinstance(out, Tensor)]

            if output_tensors:
                grad_outputs: list[_NumPyArray | None] = [None] * len(output_tensors)
                called = False

                def _call_full_backward_hooks() -> None:
                    nonlocal called, grad_outputs
                    if called:
                        return
                    called = True

                    grad_output_tuple = tuple(grad_outputs)
                    for hook in self._full_backward_pre_hooks:
                        result = hook(self, grad_output_tuple)
                        if result is not None:
                            grad_output_tuple = result

                    grad_input_tuple = tuple(
                        arg.grad if isinstance(arg, Tensor) else None for arg in args
                    )
                    for hook in self._full_backward_hooks:
                        hook(self, grad_input_tuple, grad_output_tuple)

                for idx, out in enumerate(output_tensors):

                    def _make_hook(index: int) -> Callable:
                        def _hook(_, grad: _NumPyArray) -> None:
                            grad_outputs[index] = grad
                            if all(g is not None for g in grad_outputs):
                                _call_full_backward_hooks()

                        return _hook

                    out.register_hook(_make_hook(idx))

        return output

    def __repr__(self) -> str:
        extra = self.extra_repr()
        child_lines = []
        for name, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)
            child_lines.append(f"({name}): {mod_str}")

        main_str = self._get_name() + "("
        if extra:
            main_str += extra
        if child_lines:
            if extra:
                main_str += "\n"
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str

    def extra_repr(self) -> str:
        exclude = {"training", "device"}
        attrs = []
        for name, value in vars(self).items():
            if name.startswith("_") or name in exclude:
                continue
            if (
                name in self._parameters
                or name in self._buffers
                or name in self._modules
            ):
                continue
            attrs.append(f"{name}={value}")
        return ", ".join(attrs)

    def _get_name(self) -> str:
        return self._alt_name or type(self).__name__


def _add_indent(s: str, num_spaces: int) -> str:
    lines = s.splitlines()
    if len(lines) <= 1:
        return s
    first = lines[0]
    indented = [" " * num_spaces + line for line in lines[1:]]
    return "\n".join([first, *indented])


T = TypeVar("T", bound=Type[Module])


def auto_repr(*attr_names: str) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
        def extra_repr(self: Module) -> str:
            parts = []
            for name in attr_names:
                val = getattr(self, name, None)
                parts.append(f"{name}={val}")
            return ", ".join(parts)

        cls.extra_repr = extra_repr
        return cls

    return wrapper


def set_state_dict_pass_attr(*attr_names: str) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
        cls._state_dict_pass_attr = set(attr_names)
        return cls

    return wrapper


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

    def __repr__(self) -> str:
        lines = []
        for i, (_, p) in enumerate(self._parameters.items()):
            param_str = _add_indent(repr(p), 2)
            lines.append(f"({i}): {param_str}")

        main_str = self._get_name() + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


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

    def __repr__(self) -> str:
        lines = []
        for name, param in self._parameters.items():
            param_str = _add_indent(repr(param), 2)
            lines.append(f"({name}): {param_str}")

        main_str = self._get_name() + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str
