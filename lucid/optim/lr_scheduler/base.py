from typing import Any
from abc import ABC, abstractmethod

from lucid.optim import Optimizer


class LRScheduler(ABC):
    def __init__(
        self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
    ) -> None:
        super().__init__()
        if not hasattr(optimizer, "param_groups"):
            raise TypeError(f"{type(optimizer).__name__} is not a valid optimizer.")

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        self._step_count = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    @abstractmethod
    def get_lr(self) -> list[float]:
        raise NotImplementedError("get_lr must be implemented in subclasses.")

    def step(self, epoch: int | None = None) -> None:
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self._step_count += 1
            self.last_epoch = self._step_count

        new_lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = lr

        self._last_lr = new_lrs

        if self.verbose:
            print(f"Epoch {self.last_epoch}: setting learning rates to {new_lrs}.")

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_epoch": self.last_epoch,
            "base_lrs": self.base_lrs,
            "_step_count": self._step_count,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_epoch = state_dict["last_epoch"]
        self.base_lrs = state_dict["base_lrs"]

        self._step_count = state_dict["_step_count"]
        self._last_lr = state_dict["_last_lr"]

        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group["lr"] = lr

    @property
    def last_lr(self) -> list[float]:
        return self._last_lr
