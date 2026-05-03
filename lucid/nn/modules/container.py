"""
Container modules: Sequential, ModuleList, ModuleDict.
"""

from collections import OrderedDict
from typing import Any, Iterator, overload
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._tensor.tensor import Tensor


class Sequential(Module):
    """Sequentially apply a list of modules."""

    @overload
    def __init__(self, *args: Module) -> None: ...
    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x: Any) -> Any:
        for m in self._modules.values():
            x = m(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        keys = list(self._modules.keys())
        return self._modules[keys[idx]]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        yield from self._modules.values()

    def append(self, module: Module) -> "Sequential":
        self.add_module(str(len(self._modules)), module)
        return self


class ModuleList(Module):
    """Hold submodules in a list."""

    def __init__(self, modules: list[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                self.add_module(str(i), m)

    def __getitem__(self, idx: int) -> Module:
        keys = list(self._modules.keys())
        return self._modules[keys[idx]]

    def __setitem__(self, idx: int, module: Module) -> None:
        keys = list(self._modules.keys())
        self._modules[keys[idx]] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        yield from self._modules.values()

    def append(self, module: Module) -> "ModuleList":
        self.add_module(str(len(self._modules)), module)
        return self

    def forward(self, *args: Any) -> Any:
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

    def forward(self, *args: Any) -> Any:
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

    def __len__(self) -> int:
        return len(self._parameters)

    def append(self, param: Parameter) -> "ParameterList":
        self.register_parameter(str(len(self._parameters)), param)
        return self

    def forward(self, *args: Any) -> Any:
        raise NotImplementedError


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

    def forward(self, *args: Any) -> Any:
        raise NotImplementedError
