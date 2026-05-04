"""
Learning rate schedulers.
"""

import math
from typing import Callable
from lucid.optim.optimizer import Optimizer


class _LRScheduler:
    """Base LR scheduler."""

    def __init__(
        self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
    ) -> None:
        self.optimizer = optimizer
        self.last_epoch = (
            last_epoch + 1
        )  # starts at 0; after N step() calls, last_epoch == N
        self._step_count = 0
        self.verbose = verbose
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._last_lr: list[float] = list(self.base_lrs)

    def step(self) -> None:
        """Advance the scheduler by one epoch and update learning rates."""
        self.last_epoch += 1
        self._step_count += 1
        values = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = lr
        self._last_lr = list(values)
        self.optimizer._sync_hyperparams()
        if self.verbose:
            self.print_lr(self.verbose, self.last_epoch, values)

    def get_lr(self) -> list[float]:
        """Compute new LRs. Override in subclasses."""
        raise NotImplementedError

    def get_last_lr(self) -> list[float]:
        """Return the last computed LR per group (optimizer's current LR)."""
        return [g["lr"] for g in self.optimizer.param_groups]

    def print_lr(self, is_verbose: bool, epoch: int, lrs: list[float]) -> None:
        """Print the current learning rates if verbose is enabled."""
        if is_verbose:
            for i, lr in enumerate(lrs):
                print(
                    f"Epoch {epoch}: adjusting learning rate of group {i} to {lr:.4e}."
                )


class StepLR(_LRScheduler):
    """Decay LR by gamma every step_size epochs."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [lr * self.gamma for lr in self.get_last_lr()]


class ExponentialLR(_LRScheduler):
    """Decay LR by gamma every epoch."""

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        return [lr * self.gamma for lr in self.get_last_lr()]


class MultiStepLR(_LRScheduler):
    """Decay LR by gamma at specified milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch not in self.milestones:
            return [g["lr"] for g in self.optimizer.param_groups]
        return [lr * self.gamma for lr in self.get_last_lr()]


class CosineAnnealingLR(_LRScheduler):
    """Cosine annealing schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
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


class LambdaLR(_LRScheduler):
    """LR determined by a user-defined function."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | list[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        return [
            base_lr * fn(self.last_epoch)
            for base_lr, fn in zip(self.base_lrs, self.lr_lambdas)
        ]


class CyclicLR(_LRScheduler):
    """Cyclic learning rate policy."""

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        mode: str = "triangular",
        gamma: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.base_lr_val = base_lr
        self.max_lr_val = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        scale = max(0.0, 1.0 - x)
        if self.mode == "triangular2":
            scale /= 2 ** (cycle - 1)
        elif self.mode == "exp_range":
            scale *= self.gamma**self.last_epoch
        lr = self.base_lr_val + (self.max_lr_val - self.base_lr_val) * scale
        return [lr] * len(self.optimizer.param_groups)


class ReduceLROnPlateau:
    """Reduce LR when a metric has stopped improving."""

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold: float = 1e-4,
        min_lr: float = 0,
    ) -> None:
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

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        step = max(1, self.last_epoch)
        scale = self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
        return [scale] * len(self.optimizer.param_groups)


class MultiplicativeLR(_LRScheduler):
    """Multiply LR by a factor returned by lr_lambda each epoch."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float],
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch == 0:
            return [g["lr"] for g in self.optimizer.param_groups]
        factor = self.lr_lambda(self.last_epoch)
        return [g["lr"] * factor for g in self.optimizer.param_groups]


class LinearLR(_LRScheduler):
    """Linearly interpolate LR from start_factor to end_factor over total_iters steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        t = min(self.last_epoch, self.total_iters)
        factor = (
            self.start_factor
            + (self.end_factor - self.start_factor) * t / self.total_iters
        )
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(_LRScheduler):
    """Scale LR by factor for total_iters steps, then restore to base LR."""

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.total_iters:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        return list(self.base_lrs)


class PolynomialLR(_LRScheduler):
    """Decay LR polynomially over total_iters steps, reaching eta_min."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.total_iters = total_iters
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> list[float]:
        if self.last_epoch >= self.total_iters:
            return [self.eta_min] * len(self.base_lrs)
        t = self.last_epoch
        T = self.total_iters
        return [
            (base_lr - self.eta_min) * ((1 - t / T) ** self.power) + self.eta_min
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing with warm restarts (SGDR)."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self._T_cur = 0
        self._T_i = T_0
        super().__init__(optimizer, last_epoch, verbose)

    def step(self) -> None:
        self.last_epoch += 1
        self._step_count += 1
        self._T_cur += 1
        values = self.get_lr()  # compute before reset so T_cur==T_i yields eta_min
        if self._T_cur >= self._T_i:
            self._T_cur = 0
            self._T_i *= self.T_mult
        for group, lr in zip(self.optimizer.param_groups, values):
            group["lr"] = lr
        self._last_lr = list(values)
        self.optimizer._sync_hyperparams()
        if self.verbose:
            self.print_lr(self.verbose, self.last_epoch, values)

    def get_lr(self) -> list[float]:
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self._T_cur / self._T_i))
            / 2
            for base_lr in self.base_lrs
        ]


class OneCycleLR(_LRScheduler):
    """1cycle learning rate policy."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch, verbose)

    def _annealing_cos(self, start: float, end: float, pct: float) -> float:
        return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)

    def _annealing_linear(self, start: float, end: float, pct: float) -> float:
        return start + pct * (end - start)

    def get_lr(self) -> list[float]:
        anneal = (
            self._annealing_cos
            if self.anneal_strategy == "cos"
            else self._annealing_linear
        )
        t = self.last_epoch
        T = self.total_steps
        warmup_steps = int(T * self.pct_start)
        results = []
        for base_lr in self.base_lrs:
            init_lr = base_lr / self.div_factor
            min_lr = init_lr / self.final_div_factor
            if t <= warmup_steps:
                lr = anneal(init_lr, self.max_lr, t / max(1, warmup_steps))
            else:
                lr = anneal(
                    self.max_lr, min_lr, (t - warmup_steps) / max(1, T - warmup_steps)
                )
            results.append(lr)
        return results


class SequentialLR:
    """Run a sequence of schedulers, switching at specified milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: list[_LRScheduler],
        milestones: list[int],
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = last_epoch
        self._idx = 0

    def step(self) -> None:
        self.last_epoch += 1
        while (
            self._idx < len(self.milestones)
            and self.last_epoch >= self.milestones[self._idx]
        ):
            self._idx += 1
        self.schedulers[min(self._idx, len(self.schedulers) - 1)].step()

    def get_last_lr(self) -> list[float]:
        return self.schedulers[min(self._idx, len(self.schedulers) - 1)].get_last_lr()


class ChainedScheduler:
    """Apply a list of schedulers sequentially each step (product of their LR changes)."""

    def __init__(self, schedulers: list[_LRScheduler]) -> None:
        self.schedulers = schedulers

    def step(self) -> None:
        for sched in self.schedulers:
            sched.step()

    def get_last_lr(self) -> list[float]:
        return self.schedulers[-1].get_last_lr()
