from collections import defaultdict
from typing import Any, Iterable
from abc import ABC, abstractmethod
import copy

import lucid.nn as nn

from lucid.types import _OptimClosure


class Optimizer(ABC):
    def __init__(
        self, params: Iterable[nn.Parameter], defaults: dict[str, Any]
    ) -> None:
        if not isinstance(params, Iterable):
            raise TypeError("params should be an iterable of Parameters.")

        param_list = list(params)
        for p in param_list:
            if not isinstance(p, nn.Parameter):
                raise TypeError(f"Expected nn.Parameter, got {type(p).__name__}.")

        self.defaults: dict[str, Any] = dict(defaults)
        self.param_groups: list[dict[str, Any]] = self.param_groups_setup(
            param_list, self.defaults
        )
        self.state: dict[nn.Parameter, dict[str, Any]] = defaultdict(dict)

    @abstractmethod
    def step(self, closure: _OptimClosure | None = None) -> Any | None:
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, nn.Parameter):
                    param.zero_grad()

    def param_groups_setup(
        self, params: list[nn.Parameter], defaults: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return [{"params": list(params), **defaults}]

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if "params" not in param_group:
            raise ValueError("param_group must have a 'params' key.")

        params = list(param_group["params"])
        if len(params) == 0:
            raise ValueError("param_group['params'] must be non-empty.")

        for p in params:
            if not isinstance(p, nn.Parameter):
                raise TypeError(
                    f"Expected nn.Parameter in param_group, got {type(p).__name__}."
                )

        existing = set()
        for g in self.param_groups:
            existing.update(g["params"])

        if any(p in existing for p in params):
            raise ValueError("Some parameters appear in more than one parameter group.")

        filled = {
            **self.defaults,
            **{k: v for k, v in param_group.items() if k != "params"},
        }
        filled["params"] = params
        self.param_groups.append(filled)

    def _flat_params(self) -> list[nn.Parameter]:
        flat: list[nn.Parameter] = []
        for g in self.param_groups:
            flat.extend(g["params"])

        return flat

    def state_dict(self) -> dict:
        param_to_idx: dict[nn.Parameter, int] = {}
        for idx, p in enumerate(self._flat_params()):
            if p not in param_to_idx:
                param_to_idx[p] = idx

        packed_state: dict[int, dict[str, Any]] = {}
        for p, st in self.state.items():
            if p in param_to_idx:
                packed_state[param_to_idx[p]] = copy.deepcopy(st)

        packed_groups: list[dict[str, Any]] = []
        for g in self.param_groups:
            new_g: dict[str, Any] = {}

            for k, v in g.items():
                if k == "params":
                    new_g[k] = [param_to_idx[p] for p in v]
                else:
                    new_g[k] = copy.deepcopy(v)

            packed_groups.append(new_g)

        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: dict) -> None:
        if (
            not isinstance(state_dict, dict)
            or "state" not in state_dict
            or "param_groups" not in state_dict
        ):
            raise TypeError("Invalid state_dict format for Optimizer.")

        saved_groups = state_dict["param_groups"]
        saved_state = state_dict["state"]

        current_params = self._flat_params()
        n_current = len(current_params)

        new_groups: list[dict[str, Any]] = []
        for sg in saved_groups:
            if "params" not in sg:
                raise KeyError("Saved param_group missing 'params'.")
            indices: list[int] = list(sg["params"])

            if any(i < 0 or i >= n_current for i in indices):
                raise IndexError("Saved state refers to parameter index out of range.")

            params = [current_params[i] for i in indices]
            ng = {
                k: (params if k == "params" else copy.deepcopy(v))
                for k, v in sg.items()
            }
            new_groups.append(ng)

        self.param_groups = new_groups

        self.state = defaultdict(dict)
        for i, p in enumerate(self._flat_params()):
            if i in saved_state:
                self.state[p] = copy.deepcopy(saved_state[i])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.defaults})"
