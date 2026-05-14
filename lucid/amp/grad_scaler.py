"""
GradScaler for mixed-precision training.

Pure Python implementation — no engine changes required.
"""

from typing import TYPE_CHECKING

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
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        """Initialize the scaler state.

        Parameters
        ----------
        init_scale : float, default=2**16
            Initial loss scaling factor applied by :meth:`scale`.
        growth_factor : float, default=2.0
            Multiplier applied to the scale after ``growth_interval``
            consecutive non-overflowing steps.
        backoff_factor : float, default=0.5
            Multiplier applied when an inf/NaN gradient is detected.
        growth_interval : int, default=2000
            Number of overflow-free steps required before the scale grows.
        enabled : bool, default=True
            When ``False`` the scaler is a transparent pass-through.
        """
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._growth_tracker = 0
        self._scale_seq_len: int = 0
        self._found_inf = False

    def scale(self, outputs: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        """Multiply outputs by the current scale factor.

        Args:
            outputs: A Tensor or list of Tensors to scale.

        Returns:
            Scaled Tensor(s) — same structure as input.
        """
        if not self._enabled:
            return outputs

        from lucid._tensor.tensor import Tensor

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
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap, _wrap

        inv_scale = 1.0 / self._scale
        self._found_inf = False
        for group in optimizer.param_groups:
            for p in group["params"]:  # type: ignore[attr-defined]
                if p.grad is None:
                    continue
                g_impl = _unwrap(p.grad)
                # Check for non-finite values using engine isfinite.
                fin = _C_engine.isfinite(g_impl)
                # all(fin): cast bool tensor to float, sum, compare to numel.
                one = _C_engine.full(list(fin.shape), 1.0, _C_engine.F32, g_impl.device)
                zero = _C_engine.zeros(list(fin.shape), _C_engine.F32, g_impl.device)
                fin_f = _C_engine.where(fin, one, zero)
                n_fin = float(_wrap(_C_engine.sum(fin_f, [], False)).item())
                if n_fin < g_impl.numel():
                    self._found_inf = True
                else:
                    coef = _C_engine.full(
                        list(g_impl.shape), inv_scale, g_impl.dtype, g_impl.device
                    )
                    p._impl.set_grad(_C_engine.mul(g_impl, coef))

    def step(
        self, optimizer: Optimizer, *args: object, **kwargs: object
    ) -> Tensor | None:
        """Unscale gradients and call optimizer.step() if no inf/nan detected.

        If inf/nan is detected in gradients, skip the optimizer step.

        Args:
            optimizer: The optimizer to step.

        Returns:
            The return value of optimizer.step(), or None if step was skipped.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)  # type: ignore[arg-type]

        self.unscale_(optimizer)
        if not self._found_inf:
            return optimizer.step(*args, **kwargs)  # type: ignore[arg-type]
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

    def state_dict(self) -> dict[str, float]:
        """Return serializable state dict."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        """Load state from a dict."""
        self._scale = float(state_dict["scale"])
        self._growth_factor = float(state_dict["growth_factor"])
        self._backoff_factor = float(state_dict["backoff_factor"])
        self._growth_interval = int(state_dict["growth_interval"])
        self._growth_tracker = int(state_dict["growth_tracker"])
