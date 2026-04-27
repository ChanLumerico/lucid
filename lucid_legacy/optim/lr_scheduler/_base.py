from typing import Any
from abc import ABC, abstractmethod

from lucid.optim import Optimizer


class LRScheduler(ABC):
    def __init__(
        self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
    ) -> None:
        if not hasattr(optimizer, "param_groups"):
            raise TypeError(f"{type(optimizer).__name__} is not a valid optimizer.")

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lrs: list[float] = [float(g["lr"]) for g in optimizer.param_groups]

        self._step_count = 0
        self._last_lr: list[float] = [float(g["lr"]) for g in optimizer.param_groups]

    @abstractmethod
    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._step_count += 1

        new_lrs = self.get_lr()
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                f"get_lr returned {len(new_lrs)} values, "
                f"but optimizer has {len(self.optimizer.param_groups)} param groups."
            )

        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = float(lr)

        self._last_lr = [float(g["lr"]) for g in self.optimizer.param_groups]

        if self.verbose:
            print(
                f"Epoch {self.last_epoch}: setting learning rates to {self._last_lr}."
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_epoch": int(self.last_epoch),
            "base_lrs": [float(x) for x in self.base_lrs],
            "_step_count": int(self._step_count),
            "_last_lr": [float(x) for x in self._last_lr],
            "_group_count": len(self.optimizer.param_groups),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        required = {"last_epoch", "base_lrs", "_step_count", "_last_lr"}
        missing = required - set(state_dict)
        if missing:
            raise KeyError(f"Missing keys in scheduler state_dict: {missing}")

        saved_group_count = int(
            state_dict.get("_group_count", len(state_dict["_last_lr"]))
        )
        current_group_count = len(self.optimizer.param_groups)
        if saved_group_count != current_group_count:
            raise ValueError(
                "Cannot load scheduler state: param group count mismatch "
                f"(saved={saved_group_count}, current={current_group_count})."
            )

        self.last_epoch = int(state_dict["last_epoch"])
        self.base_lrs = [float(x) for x in state_dict["base_lrs"]]
        self._step_count = int(state_dict["_step_count"])
        self._last_lr = [float(x) for x in state_dict["_last_lr"]]

        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group["lr"] = float(lr)

    @property
    def last_lr(self) -> list[float]:
        return list(self._last_lr)
