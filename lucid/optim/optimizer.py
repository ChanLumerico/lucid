"""
Optimizer base class.
"""

from typing import Iterable

from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid.nn.parameter import Parameter

from lucid._C import engine as _C_engine


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

            def _step_with_eval(
                self: Optimizer, closure: _OptimizerClosure = None
            ) -> Tensor | None:
                result = _orig(self, closure)
                self._metal_eval_params()
                return result

            _step_with_eval.__name__ = "step"
            _step_with_eval.__doc__ = _orig.__doc__
            cls.step = _step_with_eval  # type: ignore[method-assign]

    def _metal_eval_params(self) -> None:
        """Flush all parameter tensors via C++ eval_tensors() — no mlx import.

        Called automatically after every optimizer step.
        GPU tensors are flushed in one C++ call; CPU tensors are ignored.
        """

        impls: list[object] = [
            p._impl
            for group in self.param_groups
            for p in group["params"]
            if isinstance(p, Parameter)
        ]
        if impls:
            _C_engine.eval_tensors(impls)  # C++ handles GPU/CPU filtering

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
        self.state: dict[int, dict[str, object]] = {}
        self.defaults: dict[str, object] = defaults

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

    # ── state_dict round-trip ─────────────────────────────────────────────────
    #
    # Format follows reference framework: ``param_groups`` mirrors the live
    # groups but each ``params`` entry is replaced with a list of integer
    # parameter ids; ``state`` is keyed by those same ids. Parameter tensors
    # themselves live on the model — saving them inside the optimizer would
    # double-checkpoint the weights and break partial restores.
    #
    # Engine optimizers (Adam, AdamW, SGD) expose their per-parameter mutable
    # state via the C++ ``state_buffers``/``load_state_buffers`` hooks; the base
    # class harvests those automatically below. Subclasses that own additional
    # Python-side state (e.g. LBFGS history) should override _save_state /
    # _load_state to round-trip it.

    def _save_state(self) -> dict[int, dict[str, object]]:
        """Snapshot per-parameter state. Default: pull from engine optimizers."""
        return self._save_engine_state()

    def _load_state(self, state: dict[int, dict[str, object]]) -> None:
        """Restore per-parameter state. Default: push back to engine optimizers."""
        self._load_engine_state(state)

    def _save_engine_state(self) -> dict[int, dict[str, object]]:
        """Snapshot every engine optimizer's state buffers + step_count.

        Output is keyed by flat parameter index. Each entry stores:
        - one numpy array per state buffer (``exp_avg``, ``exp_avg_sq``,
          ``momentum_buffer`` ...) keyed by buffer name
        - ``step``: per-group step counter (broadcast across all params in
          that group, so it's available wherever you look it up)
        """
        out: dict[int, dict[str, object]] = {}
        flat_idx: int = 0
        for group, eng in zip(self.param_groups, self._engine_optims):
            params: list[Parameter] = group["params"]  # type: ignore[assignment]
            if eng is None:
                flat_idx += len(params)
                continue
            buffers: list[tuple[str, list[object]]] = eng.state_buffers()
            step_count: int = int(getattr(eng, "step_count", 0) or 0)
            for slot, _ in enumerate(params):
                snapshot: dict[str, object] = {}
                for name, tensors in buffers:
                    if slot < len(tensors) and tensors[slot] is not None:
                        # tensors[slot] is a TensorImpl — round-trip via numpy
                        # so the saved checkpoint is portable across processes
                        # (no shared C++ pointers across pickling).
                        import numpy as _np

                        snapshot[name] = _np.asarray(
                            tensors[slot].data_as_python()
                        ).copy()
                if step_count != 0:
                    snapshot["step"] = step_count
                if snapshot:
                    out[flat_idx + slot] = snapshot
            flat_idx += len(params)
        return out

    def _load_engine_state(self, state: dict[int, dict[str, object]]) -> None:
        """Push numpy-backed state buffers back into each engine optimizer."""
        if not state:
            return
        flat_idx: int = 0
        for group, eng in zip(self.param_groups, self._engine_optims):
            params: list[Parameter] = group["params"]  # type: ignore[assignment]
            if eng is None:
                flat_idx += len(params)
                continue
            # Collect per-buffer-name lists running parallel to params.
            by_name: dict[str, list[object | None]] = {}
            step_count: int = 0
            for slot, p in enumerate(params):
                snapshot: dict[str, object] = state.get(flat_idx + slot, {})  # type: ignore[arg-type]
                for k, v in snapshot.items():
                    if k == "step":
                        step_count = max(step_count, int(v))  # type: ignore[arg-type]
                        continue
                    by_name.setdefault(k, [None] * len(params))
                    # Wrap as TensorImpl on the param's device so the engine
                    # can copy it back into its buffer slot.
                    by_name[k][slot] = _C_engine.TensorImpl(v, p._impl.device, False)
            if by_name:
                eng.load_state_buffers([(k, v) for k, v in by_name.items()])
            if step_count and hasattr(eng, "step_count"):
                eng.step_count = step_count
            flat_idx += len(params)

    def _param_id_map(self) -> dict[int, int]:
        """Map ``id(param)`` → flat integer index across all param groups."""
        out: dict[int, int] = {}
        idx: int = 0
        for group in self.param_groups:
            for p in group["params"]:
                out[id(p)] = idx
                idx += 1
        return out

    def state_dict(self) -> dict[str, object]:
        """Return a checkpointable snapshot of the optimizer.

        Mirrors reference framework's optimizer state_dict layout:

        - ``state``: ``{param_index: {key: value, ...}}`` — Python-side per-
          parameter state (e.g. LBFGS history). Engine-managed moments (Adam)
          are not currently captured.
        - ``param_groups``: list of group dicts; each group's ``params`` is a
          list of integer indices into the flat parameter list.
        """
        id_map: dict[int, int] = self._param_id_map()
        groups_out: list[dict[str, object]] = []
        for group in self.param_groups:
            g: dict[str, object] = {k: v for k, v in group.items() if k != "params"}
            g["params"] = [id_map[id(p)] for p in group["params"]]
            groups_out.append(g)
        self.state = self._save_state()
        return {"state": self.state, "param_groups": groups_out}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore from a state_dict produced by :meth:`state_dict`.

        Hyperparameters in ``param_groups`` are restored. Python-side state
        (returned from :meth:`_save_state`) is restored via :meth:`_load_state`.
        Engine-managed moment buffers are not restored — see class docstring.
        """
        loaded_groups: list[dict[str, object]] = state_dict["param_groups"]  # type: ignore[assignment]
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state_dict has {len(loaded_groups)} param_groups but "
                f"optimizer has {len(self.param_groups)}"
            )
        for g_new, g_old in zip(self.param_groups, loaded_groups):
            for k, v in g_old.items():
                if k == "params":
                    continue
                g_new[k] = v
        loaded_state: dict[int, dict[str, object]] = state_dict.get("state", {})  # type: ignore[assignment]
        self.state = loaded_state
        self._load_state(loaded_state)
        self._sync_hyperparams()
