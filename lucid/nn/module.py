"""
lucid.nn.module — Module and container classes.

Module is the foundation of all `lucid.nn` building blocks. It manages
three OrderedDict registries — `_parameters`, `_buffers`, `_modules` —
populated transparently by `__setattr__`. State dict, train/eval,
device movement, and forward/backward hooks all delegate through
those registries.

Containers built on Module:
  * Sequential, ModuleList, ModuleDict
  * ParameterList, ParameterDict
"""

from __future__ import annotations

from typing import (
    Any, Callable, ItemsView, Iterator, KeysView,
    Self, Type, TypeVar, ValuesView, overload,
)
from collections import OrderedDict

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from lucid._tensor import Tensor
from lucid.nn.parameter import Parameter, Buffer
from lucid.types import (
    _ArrayOrScalar, _DeviceType,
    _BackwardHook, _ForwardHook, _ForwardHookKwargs,
    _ForwardPreHook, _ForwardPreHookKwargs,
    _FullBackwardHook, _FullBackwardPreHook,
    _LoadStateDictPostHook, _LoadStateDictPreHook,
    _StateDictHook, _StateDictPreHook,
)


__all__ = [
    "Module",
    "Sequential", "ModuleList", "ModuleDict",
    "ParameterList", "ParameterDict",
    "auto_repr", "set_state_dict_pass_attr",
]


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #

class Module:
    _alt_name: str = ""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())

        self.training: bool = True
        self.device: _DeviceType = "cpu"

        self._forward_pre_hooks: list[
            tuple[_ForwardPreHook | _ForwardPreHookKwargs, bool]
        ] = []
        self._forward_hooks: list[
            tuple[_ForwardHook | _ForwardHookKwargs, bool]
        ] = []

        self._backward_hooks: list[_BackwardHook] = []
        self._full_backward_pre_hooks: list[_FullBackwardPreHook] = []
        self._full_backward_hooks: list[_FullBackwardHook] = []

        self._state_dict_pre_hooks: list[_StateDictPreHook] = []
        self._state_dict_hooks: list[_StateDictHook] = []

        self._load_state_dict_pre_hooks: list[_LoadStateDictPreHook] = []
        self._load_state_dict_post_hooks: list[_LoadStateDictPostHook] = []

        self._state_dict_pass_attr: set[str] = set()

    # --- Attribute routing -------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        registry_map = {
            Parameter: getattr(self, "_parameters", None),
            Buffer:    getattr(self, "_buffers", None),
            Module:    getattr(self, "_modules", None),
        }

        target = None
        for cls, registry in registry_map.items():
            if registry is not None and isinstance(value, cls):
                target = registry
                break

        if target is not None:
            for registry in registry_map.values():
                if registry is not None and registry is not target and name in registry:
                    del registry[name]
            target[name] = value
        else:
            for registry in registry_map.values():
                if registry is not None and name in registry:
                    del registry[name]

        super().__setattr__(name, value)

    def setattr_raw(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    # --- Registration helpers ---------------------------------------------

    def add_module(self, name: str, module: "Module | None") -> None:
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module.")
        self.__setattr__(name, module)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        if not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"{param} is not a Parameter.")
        self.__setattr__(name, param)

    def register_buffer(
        self,
        name: str,
        buffer: Buffer | _ArrayOrScalar | None,
        dtype: type | None = None,
    ) -> None:
        if buffer is not None and not isinstance(buffer, Buffer):
            buffer = Buffer(buffer, dtype=dtype, device=self.device)
        self.__setattr__(name, buffer)

    # --- Hook registration -------------------------------------------------

    def register_forward_pre_hook(
        self, hook, *, with_kwargs: bool = False
    ) -> Callable:
        self._forward_pre_hooks.append((hook, with_kwargs))
        return lambda: self._forward_pre_hooks.remove((hook, with_kwargs))

    def register_forward_hook(
        self, hook, *, with_kwargs: bool = False
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

    # --- Defaults ----------------------------------------------------------

    def reset_parameters(self) -> None:
        for param in self.parameters():
            param.zero_()

    def forward(self, *args, **kwargs) -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError(
            "The forward method must be implemented by the subclass."
        )

    # --- Mode + device -----------------------------------------------------

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
            if module is self:
                continue
            module.to(device)
        return self

    # --- Iteration ---------------------------------------------------------

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        seen: set[int] = set()

        def _iter(mod: "Module") -> Iterator[Parameter]:
            for p in mod._parameters.values():
                if p is None:
                    continue
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                yield p
            if recurse:
                for child in mod._modules.values():
                    yield from _iter(child)
        yield from _iter(self)

    def buffers(self, recurse: bool = True) -> Iterator[Buffer]:
        for b in self._buffers.values():
            if b is None:
                continue
            yield b
        if recurse:
            for m in self._modules.values():
                yield from m.buffers(recurse=recurse)

    def modules(self) -> Iterator[Self]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def children(self) -> Iterator[Self]:
        return iter(self._modules.values())

    def count_parameters(self, recurse: bool = True) -> int:
        return sum(p.size for p in self.parameters(recurse=recurse))

    @property
    def parameter_size(self) -> int:
        return self.count_parameters(recurse=True)

    def apply(self, fn: Callable[[Self], None]) -> Self:
        fn(self)
        for m in self._modules.values():
            m.apply(fn)
        return self

    # --- State dict --------------------------------------------------------

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

        for name, p in self._parameters.items():
            if p is None:
                continue
            destination[prefix + name] = p if keep_vars else p.numpy()

        for name, b in self._buffers.items():
            if b is None:
                continue
            destination[prefix + name] = b if keep_vars else b.numpy()

        for name, m in self._modules.items():
            m.state_dict(
                destination=destination,
                prefix=prefix + name + ".",
                keep_vars=keep_vars,
            )

        for key in list(destination.keys()):
            if key in self._state_dict_pass_attr:
                del destination[key]

        for hook in self._state_dict_hooks:
            hook(self, destination, prefix, keep_vars)

        return destination

    def load_state_dict(
        self,
        state_dict: OrderedDict,
        strict: bool = True,
        *,
        verbose: bool = False,
        progress_desc: str | None = None,
    ) -> None:
        for hook in self._load_state_dict_pre_hooks:
            hook(self, state_dict, strict)

        own_state = self.state_dict(keep_vars=True)
        missing = set(own_state.keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(own_state.keys())

        if strict:
            msg = ""
            if missing:
                msg += f"Missing keys in state_dict: {missing}\n"
            if unexpected:
                msg += f"Unexpected keys in state_dict: {unexpected}\n"
            if msg:
                raise KeyError("Error(s) in loading state_dict:\n" + msg)

        keys_to_migrate = [k for k in state_dict.keys() if k in own_state]
        total = len(keys_to_migrate)
        migrated = 0

        use_progress = verbose and tqdm is not None and total > 0
        pbar = (
            tqdm(
                total=total,
                desc=progress_desc or "Applying state-dict",
                unit="param",
                leave=False,
            )
            if use_progress
            else None
        )

        try:
            for key in keys_to_migrate:
                value = state_dict[key]
                attr = own_state[key]
                if isinstance(attr, (Parameter, Buffer)):
                    new_t = Tensor(value, device=self.device)
                    attr._impl = new_t._impl
                else:
                    setattr(self, key, value)

                migrated += 1
                if pbar is not None:
                    shape = getattr(value, "shape", None)
                    shape_str = str(tuple(shape)) if shape is not None else "-"
                    pbar.set_postfix_str(
                        f"params={migrated}/{total}, shape={shape_str}, key={key}")
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

        for hook in self._load_state_dict_post_hooks:
            hook(self, missing, unexpected, strict)

    # --- Forward dispatch --------------------------------------------------

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

        return output

    # --- JIT ---------------------------------------------------------------

    def compile(self, **kwargs):
        from lucid._jit import JITModule
        return JITModule(self, **kwargs)

    # --- Repr --------------------------------------------------------------

    def __repr__(self) -> str:
        extra = self.extra_repr()
        child_lines = []
        for name, m in self._modules.items():
            mod_str = _add_indent(repr(m), 2)
            child_lines.append(f"({name}): {mod_str}")

        main = self._get_name() + "("
        if extra:
            main += extra
        if child_lines:
            if extra:
                main += "\n"
            main += "\n  " + "\n  ".join(child_lines) + "\n"
        main += ")"
        return main

    def extra_repr(self) -> str:
        exclude = {"training", "device"}
        attrs = []
        for name, value in vars(self).items():
            if name.startswith("_") or name in exclude:
                continue
            if (name in self._parameters
                    or name in self._buffers
                    or name in self._modules):
                continue
            attrs.append(f"{name}={value}")
        return ", ".join(attrs)

    def _get_name(self) -> str:
        return self._alt_name or type(self).__name__


# --------------------------------------------------------------------------- #
# Repr helpers and class decorators
# --------------------------------------------------------------------------- #

def _add_indent(s: str, num_spaces: int) -> str:
    lines = s.splitlines()
    if len(lines) <= 1:
        return s
    first = lines[0]
    indented = [" " * num_spaces + line for line in lines[1:]]
    return "\n".join([first, *indented])


T = TypeVar("T", bound=Type[Module])


def auto_repr(*attr_names: str) -> Callable[[T], T]:
    """Decorator: replace `extra_repr()` to dump named attributes."""

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
    """Decorator: skip the listed keys when emitting `state_dict`."""

    def wrapper(cls: T) -> T:
        cls._state_dict_pass_attr = set(attr_names)
        return cls
    return wrapper


# --------------------------------------------------------------------------- #
# Sequential
# --------------------------------------------------------------------------- #

class Sequential(Module):
    @overload
    def __init__(self, *modules: Module) -> None: ...
    @overload
    def __init__(self, ordered_dict: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args: Module | OrderedDict[str, Module]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, m in args[0].items():
                self.add_module(name, m)
        else:
            for idx, m in enumerate(args):
                self.add_module(str(idx), m)

    def forward(self, input: Tensor) -> Tensor:
        for m in self._modules.values():
            input = m(input)
        return input

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            sub = list(self._modules.items())[idx]
            return Sequential(OrderedDict(sub))
        if isinstance(idx, int):
            keys = list(self._modules.keys())
            if idx < 0:
                idx += len(keys)
            if idx < 0 or idx >= len(keys):
                raise IndexError("Index out of range")
            return self._modules[keys[idx]]
        raise TypeError(f"Invalid index type: {type(idx)}")

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
        for m in modules:
            self.append(m)

    @classmethod
    def from_ordered_dict(cls, odict: OrderedDict[str, Module]) -> "Sequential":
        return cls(odict)

    @classmethod
    def from_modules(cls, *modules: Module) -> "Sequential":
        return cls(*modules)


# --------------------------------------------------------------------------- #
# ModuleList
# --------------------------------------------------------------------------- #

class ModuleList(Module):
    def __init__(self, modules: list[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            sub = list(self._modules.items())[idx]
            ml = ModuleList()
            for i, (_, m) in enumerate(sub):
                ml.add_module(str(i), m)
            return ml
        if isinstance(idx, int):
            keys = list(self._modules.keys())
            if idx < 0:
                idx += len(keys)
            if idx < 0 or idx >= len(keys):
                raise IndexError("Index out of range.")
            return self._modules[keys[idx]]
        raise TypeError(f"Invalid index type: {type(idx)}")

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
        del items[idx]
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
            index = max(0, index + total)
        if index > total:
            index = total
        items = list(self._modules.items())
        items.insert(index, (str(index), module))
        self._modules.clear()
        for i, (_, m) in enumerate(items):
            self._modules[str(i)] = m


# --------------------------------------------------------------------------- #
# ModuleDict
# --------------------------------------------------------------------------- #

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
        m = self._modules[key]
        del self._modules[key]
        return m

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


# --------------------------------------------------------------------------- #
# ParameterList / ParameterDict
# --------------------------------------------------------------------------- #

class ParameterList(Module):
    def __init__(self, parameters: list[Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            self.extend(parameters)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            items = list(self._parameters.items())[idx]
            plist = ParameterList()
            for i, (_, p) in enumerate(items):
                plist.register_parameter(str(i), p)
            return plist
        if isinstance(idx, int):
            keys = list(self._parameters.keys())
            if idx < 0:
                idx += len(keys)
            if idx < 0 or idx >= len(keys):
                raise IndexError("Index out of range.")
            return self._parameters[keys[idx]]
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __setitem__(self, idx: int, param: Parameter) -> None:
        if not isinstance(idx, int):
            raise TypeError("Indices should be integers.")
        if not isinstance(param, Parameter):
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

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())

    def append(self, param: Parameter) -> None:
        if not isinstance(param, Parameter):
            raise TypeError("Can only append Parameter to ParameterList.")
        self.register_parameter(str(len(self._parameters)), param)

    def extend(self, parameters) -> None:
        for p in parameters:
            self.append(p)

    def insert(self, index: int, param: Parameter) -> None:
        if not isinstance(index, int):
            raise TypeError("Index should be an integer for insert.")
        if not isinstance(param, Parameter):
            raise TypeError("Can only insert Parameter into ParameterList.")
        total = len(self._parameters)
        if index < 0:
            index = max(0, index + total)
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
            lines.append(f"({i}): " + _add_indent(repr(p), 2))
        main = self._get_name() + "("
        if lines:
            main += "\n  " + "\n  ".join(lines) + "\n"
        main += ")"
        return main


class ParameterDict(Module):
    def __init__(self, parameters: dict[str, Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            self.update(parameters)

    def update(self, parameters: dict[str, Parameter]) -> None:
        for k, p in parameters.items():
            self[k] = p

    def clear(self) -> None:
        self._parameters.clear()

    def pop(self, key: str) -> Parameter:
        p = self._parameters[key]
        del self._parameters[key]
        return p

    def __getitem__(self, key: str) -> Parameter:
        return self._parameters[key]

    def __setitem__(self, key: str, param: Parameter) -> None:
        if not isinstance(param, Parameter):
            raise TypeError(f"Expected Parameter, got {type(param)}")
        self.register_parameter(key, param)

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def keys(self) -> KeysView[str]:
        return self._parameters.keys()

    def values(self) -> ValuesView[Parameter]:
        return self._parameters.values()

    def items(self) -> ItemsView[str, Parameter]:
        return self._parameters.items()

    def __repr__(self) -> str:
        lines = []
        for name, p in self._parameters.items():
            lines.append(f"({name}): " + _add_indent(repr(p), 2))
        main = self._get_name() + "("
        if lines:
            main += "\n  " + "\n  ".join(lines) + "\n"
        main += ")"
        return main
