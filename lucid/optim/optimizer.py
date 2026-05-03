"""
Optimizer base class.
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.nn.parameter import Parameter


class Optimizer:
    """
    Base class for all optimizers.

    param_groups: list of dicts, each with:
      {"params": list[Parameter], "lr": float, ...}
    """

    def __init__(
        self,
        params: list["Parameter"] | list[dict[str, Any]],
        defaults: dict[str, Any],
    ) -> None:
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups: list[dict[str, Any]] = list(params)  # type: ignore[arg-type]
        else:
            param_groups = [{"params": list(params)}]  # type: ignore[arg-type]

        self.param_groups: list[dict[str, Any]] = []
        self._engine_optims: list[Any] = []
        self.state: dict[str, Any] = {}
        self.defaults = defaults

        for group in param_groups:
            self.add_param_group(group)

    def add_param_group(self, group: dict[str, Any]) -> None:
        """Add a parameter group, creating one new engine optimizer for it."""
        merged: dict[str, Any] = {**self.defaults, **group}
        merged["params"] = list(merged["params"])
        self.param_groups.append(merged)
        self._append_engine_optim(merged)

    def _append_engine_optim(self, group: dict[str, Any]) -> None:
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

    def step(self, closure: Any = None) -> Any:
        """Perform a single optimization step."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer state as a dict."""
        return {"state": self.state, "param_groups": self.param_groups}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state."""
        self.state = state_dict["state"]
        for g_new, g_old in zip(self.param_groups, state_dict["param_groups"]):
            g_new.update({k: v for k, v in g_old.items() if k != "params"})
        self._sync_hyperparams()
