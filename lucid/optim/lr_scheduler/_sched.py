import math

from lucid.optim import Optimizer
from lucid.optim.lr_scheduler import LRScheduler


__all__ = ["StepLR", "ExponentialLR", "CosineAnnealingLR"]


class StepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be a positive integer.")

        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be a positive float.")

        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        factor = self.gamma**self.last_epoch
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if T_max <= 0:
            raise ValueError("T_max must be a positive integer.")

        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
