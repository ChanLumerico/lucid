"""
Container modules: Sequential, ModuleList, ModuleDict.
"""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from typing import Iterator, overload
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._tensor.tensor import Tensor


class Sequential(Module):
    """Sequentially apply a list of modules."""

    @overload
    def __init__(self, *args: Module) -> None: ...
    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args: Module) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x

    def __getitem__(self, idx: int | slice) -> Module | Sequential:
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            return Sequential(OrderedDict((k, self._modules[k]) for k in keys[idx]))
        return self._modules[keys[idx]]

    def __setitem__(self, idx: int, module: Module) -> None:
        keys = list(self._modules.keys())
        self._modules[keys[idx]] = module

    def __delitem__(self, idx: int | slice) -> None:
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            for key in keys[idx]:
                del self._modules[key]
        else:
            del self._modules[keys[idx]]
        self._renumber_modules()

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        yield from self._modules.values()

    def append(self, module: Module) -> Sequential:
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> Sequential:
        for module in modules:
            self.append(module)
        return self

    def insert(self, index: int, module: Module) -> Sequential:
        n = len(self._modules)
        if index < 0:
            index += n
        index = max(0, min(index, n))
        items = list(self._modules.values())
        items.insert(index, module)
        self._modules.clear()
        for i, item in enumerate(items):
            self.add_module(str(i), item)
        return self

    def _renumber_modules(self) -> None:
        items = list(self._modules.values())
        self._modules.clear()
        for i, module in enumerate(items):
            self.add_module(str(i), module)


class ModuleList(Module):
    """Hold submodules in a list."""

    def __init__(self, modules: list[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                self.add_module(str(i), m)

    def __getitem__(self, idx: int | slice) -> Module | ModuleList:
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            return ModuleList([self._modules[k] for k in keys[idx]])
        return self._modules[keys[idx]]

    def __setitem__(self, idx: int, module: Module) -> None:
        keys = list(self._modules.keys())
        self._modules[keys[idx]] = module

    def __delitem__(self, idx: int | slice) -> None:
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            for key in keys[idx]:
                del self._modules[key]
        else:
            del self._modules[keys[idx]]
        self._renumber_modules()

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        yield from self._modules.values()

    def append(self, module: Module) -> ModuleList:
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> ModuleList:
        for module in modules:
            self.append(module)
        return self

    def insert(self, index: int, module: Module) -> ModuleList:
        n = len(self._modules)
        if index < 0:
            index += n
        index = max(0, min(index, n))
        items = list(self._modules.values())
        items.insert(index, module)
        self._modules.clear()
        for i, item in enumerate(items):
            self.add_module(str(i), item)
        return self

    def _renumber_modules(self) -> None:
        items = list(self._modules.values())
        self._modules.clear()
        for i, module in enumerate(items):
            self.add_module(str(i), module)

    def forward(self, *args: object) -> Tensor:
        raise NotImplementedError("ModuleList has no forward; iterate manually.")


class ModuleDict(Module):
    """Hold submodules in a dict."""

    def __init__(self, modules: dict[str, Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            for k, m in modules.items():
                self.add_module(k, m)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        yield from self._modules.keys()

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def get(self, key: str, default: Module | None = None) -> Module | None:
        return self._modules.get(key, default)

    def pop(self, key: str) -> Module:
        return self._modules.pop(key)

    def clear(self) -> None:
        self._modules.clear()

    def update(
        self, modules: Mapping[str, Module] | Iterable[tuple[str, Module]]
    ) -> ModuleDict:
        items = modules.items() if isinstance(modules, Mapping) else modules
        for key, module in items:
            self.add_module(key, module)
        return self

    def forward(self, *args: object) -> Tensor:
        raise NotImplementedError("ModuleDict has no forward; use indexing.")


class ParameterList(Module):
    """Hold Parameters in a list."""

    def __init__(self, parameters: list[Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            for i, p in enumerate(parameters):
                self.register_parameter(str(i), p)

    def __getitem__(self, idx: int) -> Parameter:
        keys = list(self._parameters.keys())
        p = self._parameters[keys[idx]]
        assert p is not None
        return p

    def __setitem__(self, idx: int, param: Parameter) -> None:
        keys = list(self._parameters.keys())
        self._parameters[keys[idx]] = param

    def __delitem__(self, idx: int | slice) -> None:
        keys = list(self._parameters.keys())
        if isinstance(idx, slice):
            for key in keys[idx]:
                del self._parameters[key]
        else:
            del self._parameters[keys[idx]]
        self._renumber_parameters()

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            assert p is not None
            yield p

    def append(self, param: Parameter) -> ParameterList:
        self.register_parameter(str(len(self._parameters)), param)
        return self

    def extend(self, parameters: Iterable[Parameter]) -> ParameterList:
        for param in parameters:
            self.append(param)
        return self

    def _renumber_parameters(self) -> None:
        items = list(self._parameters.values())
        self._parameters.clear()
        for i, param in enumerate(items):
            self.register_parameter(str(i), param)

    def forward(self, *args: object) -> Tensor:
        raise NotImplementedError(
            "ParameterList has no forward(); it is a container for Parameters only. "
            "Access individual parameters via indexing: self.params[i]"
        )


class ParameterDict(Module):
    """Hold Parameters in a dict."""

    def __init__(self, parameters: dict[str, Parameter] | None = None) -> None:
        super().__init__()
        if parameters is not None:
            for k, p in parameters.items():
                self.register_parameter(k, p)

    def __getitem__(self, key: str) -> Parameter:
        p = self._parameters[key]
        assert p is not None
        return p

    def __setitem__(self, key: str, param: Parameter) -> None:
        self.register_parameter(key, param)

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        yield from self._parameters.keys()

    def __delitem__(self, key: str) -> None:
        del self._parameters[key]

    def keys(self):
        return self._parameters.keys()

    def items(self):
        return self._parameters.items()

    def values(self):
        return self._parameters.values()

    def get(self, key: str, default: Parameter | None = None) -> Parameter | None:
        return self._parameters.get(key, default)

    def pop(self, key: str) -> Parameter:
        p = self._parameters.pop(key)
        assert p is not None
        return p

    def clear(self) -> None:
        self._parameters.clear()

    def update(
        self, parameters: Mapping[str, Parameter] | Iterable[tuple[str, Parameter]]
    ) -> ParameterDict:
        items = parameters.items() if isinstance(parameters, Mapping) else parameters
        for key, param in items:
            self.register_parameter(key, param)
        return self

    def forward(self, *args: object) -> Tensor:
        raise NotImplementedError(
            "ParameterDict has no forward(); it is a container for Parameters only. "
            "Access individual parameters via key: self.params['key']"
        )
