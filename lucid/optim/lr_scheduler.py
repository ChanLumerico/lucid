"""
Learning rate schedulers.
"""

import math
from typing import Any, Callable
from lucid.optim.optimizer import Optimizer


class _LRScheduler:
    """Base LR scheduler."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._step_count = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        """Advance the scheduler by one epoch and update learning rates."""
        self.last_epoch += 1
        self._step_count += 1
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = lr
        self.optimizer._sync_hyperparams()

    def get_lr(self) -> list[float]:
        """Compute new LRs. Override in subclasses."""
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        """Return the last computed LR per group."""
        return [g["lr"] for g in self.optimizer.param_groups]


class StepLR(_LRScheduler):
    """Decay LR by gamma every step_size epochs."""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [lr * self.gamma for lr in self.get_last_lr()]


class ExponentialLR(_LRScheduler):
    """Decay LR by gamma every epoch."""

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        return [lr * self.gamma for lr in self.get_last_lr()]


class MultiStepLR(_LRScheduler):
    """Decay LR by gamma at specified milestones."""

    def __init__(self, optimizer: Optimizer, milestones: list[int], gamma: float = 0.1,
                 last_epoch: int = -1) -> None:
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch not in self.milestones:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [lr * self.gamma for lr in self.get_last_lr()]


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing schedule."""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0,
                 last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class LambdaLR(_LRScheduler):
    """LR determined by a user-defined function."""

    def __init__(self, optimizer: Optimizer,
                 lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
                 last_epoch: int = -1) -> None:
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [base_lr * fn(self.last_epoch)
                for base_lr, fn in zip(self.base_lrs, self.lr_lambdas)]


class CyclicLR(_LRScheduler):
    """Cyclic learning rate policy."""

    def __init__(self, optimizer: Optimizer, base_lr: float, max_lr: float,
                 step_size_up: int = 2000, mode: str = "triangular",
                 gamma: float = 1.0, last_epoch: int = -1) -> None:
        self.base_lr_val = base_lr
        self.max_lr_val = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        scale = max(0.0, 1.0 - x)
        if self.mode == "triangular2":
            scale /= 2 ** (cycle - 1)
        elif self.mode == "exp_range":
            scale *= self.gamma ** self.last_epoch
        lr = self.base_lr_val + (self.max_lr_val - self.base_lr_val) * scale
        return [lr] * len(self.optimizer.param_groups)


class ReduceLROnPlateau:
    """Reduce LR when a metric has stopped improving."""

    def __init__(self, optimizer: Optimizer, mode: str = "min", factor: float = 0.1,
                 patience: int = 10, verbose: bool = False,
                 threshold: float = 1e-4, min_lr: float = 0) -> None:
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.threshold = threshold
        self.min_lr = min_lr
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._num_bad_epochs = 0

    def step(self, metrics: float) -> None:
        """Update scheduler with current metric value."""
        if self.mode == "min":
            improved = metrics < self._best - self.threshold
        else:
            improved = metrics > self._best + self.threshold

        if improved:
            self._best = metrics
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1
            if self._num_bad_epochs >= self.patience:
                for group in self.optimizer.param_groups:
                    new_lr = max(group["lr"] * self.factor, self.min_lr)
                    group["lr"] = new_lr
                self.optimizer._sync_hyperparams()
                self._num_bad_epochs = 0


class NoamScheduler(_LRScheduler):
    """Transformer Noam learning rate schedule."""

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int,
                 last_epoch: int = -1) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
        return [scale] * len(self.optimizer.param_groups)
