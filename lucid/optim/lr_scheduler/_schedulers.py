import math
from typing import Callable, Literal

from lucid.optim import Optimizer
from lucid.optim.lr_scheduler import LRScheduler


__all__ = [
    "LambdaLR",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "CyclicLR",
]


class LambdaLR(LRScheduler):
    def __init__(
        self,
        lr_lambda: Callable[[int], float],
        optimizer: Optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if not callable(lr_lambda):
            raise TypeError(
                "lr_lambda must be a callable function "
                + "that takes an epoch index and returns a scaling factor."
            )

        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        factor = self.lr_lambda(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]


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


class MultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if not milestones:
            raise ValueError("milestones must contain at least one epoch index.")
        if sorted(milestones) != milestones:
            raise ValueError("milestones must be a sorted list of increasing integers.")

        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        factor = self.gamma ** sum(self.last_epoch >= m for m in self.milestones)
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


class ReduceLROnPlateau(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"],
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        verbose: bool = False,
    ) -> None:
        if factor >= 1.0:
            raise ValueError("factor should be < 1.0.")
        if mode not in {"min", "max"}:
            raise ValueError("mode should be in 'min' or 'max'.")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold_mode must be 'rel' or 'abs'.")

        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose

        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        super().__init__(optimizer, last_epoch=-1, verbose=verbose)

    def get_lr(self) -> list[float]:
        return self._last_lr

    def step(self, metrics: float, epoch: int | None = None) -> None:
        if epoch is not None:
            self.last_epoch = epoch

        if self.is_better(metrics):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

    def is_better(self, metrics: float) -> bool:
        if self.threshold_mode == "rel":
            threshold_val = (
                self.best * (1 - self.threshold)
                if self.mode == "min"
                else self.best * (1 + self.threshold)
            )
        else:
            threshold_val = (
                self.best - self.threshold
                if self.mode == "min"
                else self.best + self.threshold
            )
        return (
            metrics < threshold_val if self.mode == "min" else metrics > threshold_val
        )

    def _reduce_lr(self) -> None:
        new_lrs = []
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group["lr"] * self.factor, self.min_lr)
            if param_group["lr"] - new_lr > self.eps:
                param_group["lr"] = new_lr

            new_lrs.append(new_lr)

        self._last_lr = new_lrs
        if self.verbose:
            print(f"Reducing learning rate to {new_lrs}.")


class CyclicLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int,
        step_size_down: int | None = None,
        mode: Literal["triangular", "triangular2", "exp_range"] = "triangular",
        gamma: float = 1.0,
        scale_fn: Callable[[int], float] | None = None,
        cycle_momentum: bool = True,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if base_lr >= max_lr:
            raise ValueError("base_lr must be less than max_lr.")
        if mode not in {"triangular", "triangular2", "exp_range"}:
            raise ValueError("Invalid mode.")

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.cycle_momentum = cycle_momentum

        self.total_size = self.step_size_up + self.step_size_down
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        cycle = self.last_epoch // self.total_size
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle - 1)

        if self.mode == "triangular":
            scale_factor = 1.0
        elif self.mode == "triangular2":
            scale_factor = 1 / (2**cycle)
        elif self.mode == "exp_range":
            scale_factor = self.gamma**self.last_epoch
        else:
            scale_factor = self.scale_fn(cycle) if self.scale_fn else 1.0

        new_lrs = [
            self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale_factor
            for _ in self.base_lrs
        ]
        return new_lrs
