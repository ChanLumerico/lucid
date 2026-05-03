"""
nn.Module: base class for all neural network layers.
"""

from collections import OrderedDict
from typing import Any, Callable, Iterator, Self, TYPE_CHECKING

from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.nn.hooks import RemovableHandle

if TYPE_CHECKING:
    pass


class Module:
    """
    Base class for all neural network modules.

    __setattr__ routing (priority order):
    1. value._is_parameter is True → _parameters[name]
    2. isinstance(value, Module)   → _modules[name]
    3. Tensor (non-parameter)      → plain attr (use register_buffer explicitly)
    4. anything else               → plain attr
    """

    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers",    OrderedDict())
        object.__setattr__(self, "_modules",    OrderedDict())
        object.__setattr__(self, "_forward_hooks",  OrderedDict())
        object.__setattr__(self, "_backward_hooks", OrderedDict())
        object.__setattr__(self, "training", True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Override in subclasses to define the computation."""
        raise NotImplementedError(f"{type(self).__name__}.forward() not implemented")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for hook in self._forward_hooks.values():
            hook(self, args, kwargs)
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        # Remove from existing dicts before re-routing
        for d in (self._parameters, self._buffers, self._modules):
            if name in d:
                del d[name]

        if isinstance(value, Parameter) and value._is_parameter:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        p = object.__getattribute__(self, "_parameters")
        if name in p:
            return p[name]
        b = object.__getattribute__(self, "_buffers")
        if name in b:
            return b[name]
        m = object.__getattribute__(self, "_modules")
        if name in m:
            return m[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    # ── parameter / module / buffer traversal ─────────────────────────────

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield all Parameters in this module (and children if recurse=True)."""
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Parameter]]:
        """Yield (name, Parameter) pairs."""
        for name, p in self._parameters.items():
            if p is not None:
                yield (f"{prefix}.{name}" if prefix else name), p
        if recurse:
            for mname, m in self._modules.items():
                subprefix = f"{prefix}.{mname}" if prefix else mname
                yield from m.named_parameters(subprefix, recurse=True)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Yield all buffer tensors."""
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Tensor]]:
        """Yield (name, buffer) pairs."""
        for name, b in self._buffers.items():
            if b is not None:
                yield (f"{prefix}.{name}" if prefix else name), b
        if recurse:
            for mname, m in self._modules.items():
                subprefix = f"{prefix}.{mname}" if prefix else mname
                yield from m.named_buffers(subprefix, recurse=True)

    def modules(self) -> Iterator[Module]:
        """Yield this module and all submodules (depth-first)."""
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_modules(
        self, prefix: str = ""
    ) -> Iterator[tuple[str, Module]]:
        """Yield (name, module) pairs."""
        yield prefix, self
        for name, m in self._modules.items():
            subprefix = f"{prefix}.{name}" if prefix else name
            yield from m.named_modules(subprefix)

    def children(self) -> Iterator[Module]:
        """Yield direct child modules."""
        yield from self._modules.values()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Yield (name, child_module) pairs."""
        yield from self._modules.items()

    # ── registration ─────────────────────────────────────────────────────

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        """Register a Parameter under the given name."""
        if param is None:
            self._parameters[name] = None  # type: ignore[assignment]
        else:
            self._parameters[name] = param

    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        """Register a buffer tensor. Buffers appear in state_dict but not parameters()."""
        self._buffers[name] = tensor

    def add_module(self, name: str, module: Module | None) -> None:
        """Add a child module."""
        if module is None:
            self._modules[name] = None  # type: ignore[assignment]
        else:
            self._modules[name] = module

    # ── training mode ────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> Self:
        """Set this module and all children to training mode."""
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Self:
        """Set this module and all children to evaluation mode."""
        return self.train(False)

    # ── device / dtype conversion ─────────────────────────────────────────

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move/cast all parameters and buffers."""
        for name, p in list(self._parameters.items()):
            if p is not None:
                new_p = Parameter(p.to(*args, **kwargs), requires_grad=p.requires_grad)
                self._parameters[name] = new_p
        for name, b in list(self._buffers.items()):
            if b is not None:
                self._buffers[name] = b.to(*args, **kwargs)
        for m in self._modules.values():
            m.to(*args, **kwargs)
        return self

    def metal(self) -> Self:
        """Move all parameters and buffers to Apple Metal GPU."""
        return self.to("metal")

    def cpu(self) -> Self:
        """Move all parameters and buffers to CPU."""
        return self.to("cpu")

    def apply(self, fn: Callable[[Module], None]) -> Self:
        """Apply fn recursively to every submodule (including self)."""
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients of all parameters."""
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p._impl.zero_grad()

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad for all parameters."""
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    # ── state_dict ────────────────────────────────────────────────────────

    def state_dict(
        self, *, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Tensor]:
        """Return a dict mapping parameter/buffer names to tensors."""
        from lucid.nn._state_dict import _save_to_state_dict
        return _save_to_state_dict(self, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(
        self, state_dict: dict[str, Tensor], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """Load parameters from a state_dict. Returns (missing_keys, unexpected_keys)."""
        from lucid.nn._state_dict import _load_from_state_dict
        return _load_from_state_dict(self, state_dict, strict=strict)

    # ── hooks ─────────────────────────────────────────────────────────────

    def register_forward_hook(
        self, hook: Callable[..., None]
    ) -> RemovableHandle:
        """Register a hook called after forward(). Returns a RemovableHandle."""
        key = id(hook)
        self._forward_hooks[key] = hook
        return RemovableHandle(self._forward_hooks, key)

    def register_full_backward_hook(
        self, hook: Callable[..., None]
    ) -> RemovableHandle:
        """Register a backward hook. Returns a RemovableHandle."""
        key = id(hook)
        self._backward_hooks[key] = hook
        return RemovableHandle(self._backward_hooks, key)

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, m in self._modules.items():
            mod_repr = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """Override to add extra repr info (e.g. Linear shows in_features, etc.)."""
        return ""
