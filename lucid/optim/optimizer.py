"""
Optimizer base class.
"""

from typing import Iterable

from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid.nn.parameter import Parameter


class Optimizer:
    """
    Base class for all optimizers.

    param_groups: list of dicts, each with:
      {"params": list[Parameter], "lr": float, ...}
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap every concrete step() so it auto-flushes Metal params after update."""
        super().__init_subclass__(**kwargs)
        if "step" in cls.__dict__:
            _orig = cls.__dict__["step"]

            def _step_with_eval(self: "Optimizer", closure: "_OptimizerClosure" = None):
                result = _orig(self, closure)
                self._metal_eval_params()
                return result

            _step_with_eval.__name__ = "step"
            _step_with_eval.__doc__  = _orig.__doc__
            cls.step = _step_with_eval  # type: ignore[method-assign]

    def _metal_eval_params(self) -> None:
        """Flush all Metal (MLX) parameter tensors in one mx.eval() call.

        Called automatically after every optimizer step when any parameter
        lives on the GPU device.  No-op on CPU-only models.
        """
        from lucid._C import engine as _ce
        gpu_impls = [
            p._impl
            for group in self.param_groups
            for p in group["params"]
            if isinstance(p, Parameter) and p._impl.device == _ce.Device.GPU
        ]
        if gpu_impls:
            import mlx.core as mx
            mx.eval(*gpu_impls)

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        defaults: dict[str, object],
    ) -> None:
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            param_groups: list[dict[str, object]] = list(params)  # type: ignore[arg-type]
        else:
            param_groups = [{"params": list(params)}]  # type: ignore[arg-type]

        self.param_groups: list[dict[str, object]] = []
        self._engine_optims: list[object] = []
        self.state: dict[str, object] = {}
        self.defaults = defaults

        for group in param_groups:
            self.add_param_group(group)

    def add_param_group(self, group: dict[str, object]) -> None:
        """Add a parameter group, creating one new engine optimizer for it."""
        merged: dict[str, object] = {**self.defaults, **group}
        merged["params"] = list(merged["params"])
        self.param_groups.append(merged)
        self._append_engine_optim(merged)

    def _append_engine_optim(self, group: dict[str, object]) -> None:
        """Create and append one engine optimizer for a single param group.

        Override in subclasses. Base is a no-op so Optimizer can be subclassed
        without a C++ backend.
        """
        pass

    def _sync_hyperparams(self) -> None:
        """Push current param_group hyperparams into existing engine optimizers.

        Preserves all accumulated optimizer state (e.g. Adam first/second moments).
        Called by LR schedulers instead of recreating engine optimizers.
        Skips read-only engine attributes silently.
        """
        for group, eng in zip(self.param_groups, self._engine_optims):
            for k, v in group.items():
                if k == "params":
                    continue
                if not hasattr(eng, k):
                    continue
                try:
                    setattr(eng, k, v)
                except (AttributeError, TypeError):
                    pass

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients of all parameters."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p._impl.zero_grad()

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single optimization step.

        Subclasses must override this to update parameters from their
        current gradients.  Optionally accepts a *closure* that re-evaluates
        the model and returns the loss (required by some optimizers, e.g.
        LBFGS; ignored by most first-order methods).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.step() is not implemented. "
            "Subclasses of Optimizer must override step()."
        )

    def state_dict(self) -> dict[str, object]:
        """Return the optimizer state as a dict."""
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Load optimizer state."""
        self.state = state_dict["state"]
        for g_new, g_old in zip(self.param_groups, state_dict["param_groups"]):
            g_new.update({k: v for k, v in g_old.items() if k != "params"})
        self._sync_hyperparams()
