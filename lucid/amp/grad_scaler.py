"""
GradScaler for mixed-precision training.

Pure Python implementation — no engine changes required.
"""

from typing import Any, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.optim.optimizer import Optimizer


class GradScaler:
    """Scale gradients to prevent underflow in mixed-precision training.

    Usage:
        scaler = GradScaler()
        with autocast():
            loss = model(x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    Args:
        init_scale:       Initial gradient scale (default: 2**16).
        growth_factor:    Multiply scale by this after growth_interval
                          consecutive non-inf/nan steps (default: 2.0).
        backoff_factor:   Multiply scale by this when inf/nan detected (default: 0.5).
        growth_interval:  Steps of no overflow before scale is increased (default: 2000).
        enabled:          If False, GradScaler is a pass-through no-op.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        self._scale_seq_len: int = 0
        self._found_inf = False

    def scale(self, outputs: Any) -> Any:
        """Multiply outputs by the current scale factor.

        Args:
            outputs: A Tensor or list of Tensors to scale.

        Returns:
            Scaled Tensor(s) — same structure as input.
        """
        if not self._enabled:
            return outputs

        from lucid._tensor.tensor import Tensor
        import lucid

        if isinstance(outputs, Tensor):
            return outputs * self._scale
        return [o * self._scale for o in outputs]

    def unscale_(self, optimizer: Optimizer) -> None:
        """Divide gradients by the current scale in-place.

        Should be called before gradient clipping.

        Args:
            optimizer: The optimizer whose parameters' grads will be unscaled.
        """
        if not self._enabled:
            return
        import numpy as np
        inv_scale = 1.0 / self._scale
        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p._impl.grad_as_python()
                if g is None:
                    continue
                if not np.all(np.isfinite(g)):
                    self._found_inf = True
                else:
                    g[:] *= inv_scale

    def step(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> Any:
        """Unscale gradients and call optimizer.step() if no inf/nan detected.

        If inf/nan is detected in gradients, skip the optimizer step.

        Args:
            optimizer: The optimizer to step.

        Returns:
            The return value of optimizer.step(), or None if step was skipped.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        self.unscale_(optimizer)
        if not self._found_inf:
            return optimizer.step(*args, **kwargs)
        return None

    def update(self, new_scale: float | None = None) -> None:
        """Update the scale factor.

        If a scale is provided, it is set directly. Otherwise, the scale
        is grown if no overflow was found for growth_interval steps, or
        reduced if overflow was found.

        Args:
            new_scale: Explicit new scale value (optional).
        """
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
            self._growth_tracker = 0
            return

        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

        self._found_inf = False

    def get_scale(self) -> float:
        """Return the current scale factor."""
        return self._scale

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state dict."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a dict."""
        self._scale = float(state_dict["scale"])
        self._growth_factor = float(state_dict["growth_factor"])
        self._backoff_factor = float(state_dict["backoff_factor"])
        self._growth_interval = int(state_dict["growth_interval"])
        self._growth_tracker = int(state_dict["growth_tracker"])
