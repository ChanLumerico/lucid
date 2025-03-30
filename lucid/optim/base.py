from collections import defaultdict
from typing import Any, Iterable
from abc import ABC, abstractmethod
import copy

import lucid.nn as nn

from lucid.types import _StateDict, _OptimClosure


class Optimizer(ABC):
    def __init__(
        self, params: Iterable[nn.Parameter], defaults: dict[str, Any]
    ) -> None:
        super().__init__()
        if not isinstance(params, Iterable):
            raise TypeError("params should be an iterable of Parameters.")

        params = list(params)
        self.param_groups = self.param_groups_setup(params, defaults)
        self.defaults = defaults
        self.state: dict[nn.Parameter, dict[str, Any]] = defaultdict(dict)

    def param_groups_setup(
        self, params: list[nn.Parameter], defaults: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [{"params": list(params), **defaults}]

    @abstractmethod
    def step(self, closure: _OptimClosure | None = None) -> Any | None:
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                param.zero_grad()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        for group in self.param_groups:
            if set(group["params"]) & set(param_group["params"]):
                raise ValueError(
                    "Some parameters appear in more than one parameter group."
                )
        self.param_groups.append(param_group)

    def state_dict(self) -> _StateDict:
        return {
            "state": copy.deepcopy(self.state),
            "param_groups": copy.deepcopy(self.param_groups),
        }

    def load_state_dict(self, state_dict: _StateDict) -> None:
        self.state = defaultdict(dict, copy.deepcopy(state_dict["state"]))
        self.param_groups = copy.deepcopy(state_dict["param_groups"])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.param_groups})"
