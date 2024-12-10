from collections import OrderedDict
from typing import Iterator, Self, overload

from lucid._tensor import Tensor
import lucid.nn as nn


class Sequential(nn.Module):  # NOTE: **WIP**
    @overload
    def __init__(self, *modules: nn.Module) -> None: ...

    @overload
    def __init__(self, ordered_dict: OrderedDict[str, nn.Module]) -> None: ...

    def __init__(self, *args: nn.Module | OrderedDict[str, nn.Module]) -> None:
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

    def __getitem__(self, idx: int | slice) -> nn.Module | Self:
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

    def __setitem__(self, idx: int, module: nn.Module) -> None:
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

    def append(self, module: nn.Module) -> None:
        self.add_module(str(len(self._modules)), module)

    def extend(self, modules: Iterator[nn.Module]) -> None:
        for module in modules:
            self.append(module)

    @classmethod
    def from_ordered_dict(cls: type[Self], odict: OrderedDict[str, nn.Module]) -> Self:
        return cls(odict)

    @classmethod
    def from_modules(cls: type[Self], *modules: nn.Module) -> Self:
        return cls(*modules)


class ModuleList(nn.Module):
    NotImplemented


class ModuleDict(nn.Module):
    NotImplemented


class ParameterList(nn.Module):
    NotImplemented


class ParameterDict(nn.Module):
    NotImplemented
