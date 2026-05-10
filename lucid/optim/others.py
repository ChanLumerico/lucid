"""
Additional optimizers: RMSprop, Adagrad, Adadelta, Adamax, RAdam, NAdam, ASGD, Rprop.
"""

from typing import Iterable, cast
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.nn.parameter import Parameter
from lucid.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.RMSprop(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("alpha", 0.99)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
                cast(float, group.get("momentum", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single RMSprop step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        eps: float = 1e-10,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Adagrad(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("lr_decay", 0.0)),
                cast(float, group.get("weight_decay", 0.0)),
                cast(float, group.get("eps", 1e-10)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adagrad step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr, rho=rho, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Adadelta(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("rho", 0.9)),
                cast(float, group.get("eps", 1e-6)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adadelta step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Adamax(Optimizer):
    """Adamax optimizer (variant of Adam based on infinity norm)."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Adamax(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Adamax step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class RAdam(Optimizer):
    """Rectified Adam optimizer."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.RAdam(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single RAdam step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class NAdam(Optimizer):
    """Nesterov Adam optimizer."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.NAdam(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("beta1", 0.9)),
                cast(float, group.get("beta2", 0.999)),
                cast(float, group.get("eps", 1e-8)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single NAdam step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class ASGD(Optimizer):
    """Averaged stochastic gradient descent."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.ASGD(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("lambd", 1e-4)),
                cast(float, group.get("alpha", 0.75)),
                cast(float, group.get("t0", 1e6)),
                cast(float, group.get("weight_decay", 0.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single ASGD step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class Rprop(Optimizer):
    """Resilient backpropagation optimizer."""

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-2,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50),
    ) -> None:
        defaults: dict[str, object] = dict(
            lr=lr,
            eta_minus=etas[0],
            eta_plus=etas[1],
            step_min=step_sizes[0],
            step_max=step_sizes[1],
        )
        super().__init__(params, defaults)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        self._engine_optims.append(
            _C_engine.Rprop(
                [_unwrap(p) for p in group["params"]],  # type: ignore[attr-defined]
                cast(float, group["lr"]),
                cast(float, group.get("eta_minus", 0.5)),
                cast(float, group.get("eta_plus", 1.2)),
                cast(float, group.get("step_min", 1e-6)),
                cast(float, group.get("step_max", 50.0)),
            )
        )

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single Rprop step."""
        loss: Tensor | None = closure() if closure is not None else None
        for optim in self._engine_optims:
            optim.step()  # type: ignore[attr-defined]  # _EngineOptimizer.step() is correct at runtime
        return loss


class SparseAdam(Optimizer):
    """Lazy version of Adam for sparse gradients.

    Dense Adam implementation that is API-compatible with reference framework SparseAdam.
    All moment state is stored as TensorImpl and updated using engine ops,
    so the optimizer works on both CPU and GPU.
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        defaults: dict[str, object] = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        # Per-parameter moment state (TensorImpl, None before first step)
        self._step: list[int] = [0] * self._n_params()
        self._exp_avg: list = [None] * self._n_params()
        self._exp_avg_sq: list = [None] * self._n_params()

    def _n_params(self) -> int:
        return sum(len(g["params"]) for g in self.param_groups)  # type: ignore[arg-type, misc]

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        loss = closure() if closure is not None else None
        flat_idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1: float
            b2: float
            b1, b2 = group["betas"]  # type: ignore[misc]  # betas is tuple[float, float] at runtime
            eps = group["eps"]
            for p in group["params"]:  # type: ignore[attr-defined]  # params is list[Parameter] at runtime
                if p.grad is None:
                    flat_idx += 1
                    continue

                pi = p._impl
                gi = p.grad._impl
                dv = pi.device
                dt = pi.dtype
                sh = list(pi.shape)

                # Initialise moment buffers
                if self._exp_avg[flat_idx] is None:
                    self._exp_avg[flat_idx] = _C_engine.zeros(sh, dt, dv)
                    self._exp_avg_sq[flat_idx] = _C_engine.zeros(sh, dt, dv)

                self._step[flat_idx] += 1
                t = self._step[flat_idx]
                m = self._exp_avg[flat_idx]
                v = self._exp_avg_sq[flat_idx]

                def _scale(
                    tensor: _C_engine.TensorImpl, scalar: float
                ) -> _C_engine.TensorImpl:
                    return _C_engine.mul(tensor, _C_engine.full(sh, scalar, dt, dv))

                # m = b1 * m + (1 - b1) * g
                m = _C_engine.add(_scale(m, b1), _scale(gi, 1.0 - b1))
                # v = b2 * v + (1 - b2) * g^2
                g_sq = _C_engine.mul(gi, gi)
                v = _C_engine.add(_scale(v, b2), _scale(g_sq, 1.0 - b2))

                self._exp_avg[flat_idx] = m
                self._exp_avg_sq[flat_idx] = v

                # Bias correction
                bc1 = 1.0 - b1**t
                bc2 = 1.0 - b2**t
                step_size = lr * (bc2**0.5) / bc1

                # p = p - step_size * m / (sqrt(v) + eps)
                denom = _C_engine.add(
                    _C_engine.sqrt(v), _C_engine.full(sh, eps, dt, dv)  # type: ignore[arg-type]
                )
                update = _scale(_C_engine.div(m, denom), step_size)
                new_p = _C_engine.sub(pi, update)
                pi.copy_from(new_p)
                flat_idx += 1
        return loss
