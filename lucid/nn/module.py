"""
nn.Module: base class for all neural network layers.
"""

from collections import OrderedDict
from typing import Callable, ClassVar, Final, Iterator, Self, TYPE_CHECKING

from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.nn.hooks import RemovableHandle
from lucid._types import _ModuleOutput, _ForwardPreHook, _ForwardHook, _BackwardHook

# _state_dict is imported lazily inside state_dict()/load_state_dict() to
# break the Module ↔ _state_dict circular dependency.


_HOOK_ID = 0


def _next_hook_id() -> int:
    global _HOOK_ID
    _HOOK_ID += 1
    return _HOOK_ID


class Module:
    """Base class for all neural network modules.

    Every custom model should subclass this and implement :meth:`forward`.
    Submodules assigned as attributes are tracked automatically.

    Notes
    -----
    **Attribute routing**: Setting an attribute follows this priority order:

    1. If the value is a :class:`~lucid.nn.Parameter` → stored in ``_parameters``.
    2. If the value is a :class:`Module` → stored in ``_modules``.
    3. Otherwise → plain Python attribute.

    To register a non-parameter tensor (e.g. a running mean), call
    :meth:`register_buffer` explicitly.

    Examples
    --------
    >>> class MLP(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10, 20)
    ...         self.fc2 = nn.Linear(20, 1)
    ...
    ...     def forward(self, x):
    ...         return self.fc2(lucid.relu(self.fc1(x)))
    ...
    >>> model = MLP()
    >>> model(lucid.randn(4, 10)).shape
    (4, 1)
    """

    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_forward_hooks", OrderedDict())
        object.__setattr__(self, "_backward_hooks", OrderedDict())
        object.__setattr__(self, "training", True)

    def forward(self, *args: Tensor, **kwargs: object) -> _ModuleOutput:
        """Override in subclasses to define the computation."""
        raise NotImplementedError(f"{type(self).__name__}.forward() not implemented")

    def __call__(self, *args: Tensor, **kwargs: object) -> _ModuleOutput:
        # pre-hooks
        for hook in self._forward_pre_hooks.values():
            result = hook(self, args)
            if result is not None:
                args = result if isinstance(result, tuple) else (result,)
        output = self.forward(*args, **kwargs)
        # post-hooks
        for hook in self._forward_hooks.values():
            hook_out = hook(self, args, output)
            if hook_out is not None:
                output = hook_out
        return output

    def __setattr__(self, name: str, value: object) -> None:
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

    def __getattr__(self, name: str) -> Tensor | Parameter | Module:
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
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Parameter]]:
        """Yield (name, Parameter) pairs.

        Parameters
        ----------
        remove_duplicate:
            If True (default), each unique Parameter object is yielded only
            once, even if referenced by multiple attributes. Mirrors PyTorch.
        """
        seen: set[int] = set()
        for name, p in self._parameters.items():
            if p is not None:
                if remove_duplicate:
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                yield (f"{prefix}.{name}" if prefix else name), p
        if recurse:
            for mname, m in self._modules.items():
                subprefix = f"{prefix}.{mname}" if prefix else mname
                for full_name, p in m.named_parameters(
                    subprefix, recurse=True, remove_duplicate=False
                ):
                    if remove_duplicate:
                        if id(p) in seen:
                            continue
                        seen.add(id(p))
                    yield full_name, p

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

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
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

    # ── dotted-path accessors ─────────────────────────────────────────────

    def get_submodule(self, target: str) -> Module:
        """Return submodule at dotted path, e.g. 'encoder.layer.0'."""
        if not target:
            return self
        parts = target.split(".")
        mod: Module = self
        for part in parts:
            if not isinstance(mod, Module):
                raise AttributeError(f"'{type(mod).__name__}' is not a Module")
            if part not in mod._modules:
                raise AttributeError(
                    f"'{type(mod).__name__}' has no submodule '{part}'"
                )
            mod = mod._modules[part]
        return mod

    def get_parameter(self, target: str) -> Parameter:
        """Return parameter at dotted path, e.g. 'fc.weight'."""
        parts = target.split(".")
        mod = self.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else self
        name = parts[-1]
        if name not in mod._parameters:
            raise AttributeError(f"'{type(mod).__name__}' has no parameter '{name}'")
        p = mod._parameters[name]
        if p is None:
            raise AttributeError(f"Parameter '{target}' is None")
        return p

    def get_buffer(self, target: str) -> Tensor:
        """Return buffer at dotted path, e.g. 'bn.running_mean'."""
        parts = target.split(".")
        mod = self.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else self
        name = parts[-1]
        if name not in mod._buffers:
            raise AttributeError(f"'{type(mod).__name__}' has no buffer '{name}'")
        b = mod._buffers[name]
        if b is None:
            raise AttributeError(f"Buffer '{target}' is None")
        return b

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
        """Register a buffer tensor. Non-persistent buffers are excluded from state_dict."""
        if not persistent:
            # Store in a separate set of non-persistent buffer names
            if not hasattr(self, "_non_persistent_buffers"):
                object.__setattr__(self, "_non_persistent_buffers", set())
            self._non_persistent_buffers.add(name)  # type: ignore[attr-defined]
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
        if not isinstance(mode, bool):
            raise TypeError(f"train() requires a bool, got {type(mode).__name__}")
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> Self:
        """Set this module and all children to evaluation mode."""
        return self.train(False)

    # ── device / dtype conversion via _apply ─────────────────────────────

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> Self:
        """Apply fn to every parameter/buffer in-place (preserves object identity)."""
        for module in self.children():
            module._apply(fn)
        for key, param in self._parameters.items():
            if param is None:
                continue
            new_impl = fn(param)._impl
            param._impl = new_impl  # mutate in-place — keeps Python object identity
        for key, buf in self._buffers.items():
            if buf is None:
                continue
            self._buffers[key] = fn(buf)
        return self

    def to(self, *args: object, **kwargs: object) -> Self:
        """Move/cast all parameters and buffers, preserving Parameter object identity."""

        def _convert(t: Tensor) -> Tensor:
            return t.to(*args, **kwargs)

        return self._apply(_convert)

    def metal(self) -> Self:
        """Move all parameters and buffers to Apple Metal GPU."""
        return self.to("metal")

    def cpu(self) -> Self:
        """Move all parameters and buffers to CPU."""
        return self.to("cpu")

    def half(self) -> Self:
        """Cast all parameters and buffers to float16."""
        from lucid._dtype import float16

        return self.to(float16)

    def float(self) -> Self:
        """Cast all parameters and buffers to float32."""
        from lucid._dtype import float32

        return self.to(float32)

    def double(self) -> Self:
        """Cast all parameters and buffers to float64."""
        from lucid._dtype import float64

        return self.to(float64)

    def bfloat16(self) -> Self:
        """Cast all parameters and buffers to bfloat16."""
        from lucid._dtype import bfloat16

        return self.to(bfloat16)

    def type(self, dst_type: object) -> Self:
        """Cast all parameters and buffers to *dst_type*.

        *dst_type* may be a :class:`lucid.dtype`, a Python type (``float``,
        ``int``), or a string (``"float32"``, ``"float16"``, etc.).
        Delegates to :meth:`to`, which handles the conversion.
        """
        return self.to(dst_type)  # type: ignore[arg-type]

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

    def share_memory(self) -> Self:
        """No-op on Apple Silicon (unified memory is always shared)."""
        return self

    def compile(self, *args: object, **kwargs: object) -> None:
        """Not implemented: JIT compilation is out of scope for this release."""
        raise NotImplementedError(
            "compile() is not available in this release. "
            "JIT compilation is planned for a future major version."
        )

    # ── extra state ───────────────────────────────────────────────────────

    def get_extra_state(self) -> object:
        """Return extra state to include in state_dict. Override in subclasses."""
        return None

    def set_extra_state(self, state: object) -> None:
        """Restore extra state loaded from state_dict. Override in subclasses."""
        pass

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

    def register_forward_pre_hook(self, hook: _ForwardPreHook) -> RemovableHandle:
        """Register a hook called before forward(). Signature: hook(module, args) -> args | None."""
        key = _next_hook_id()
        self._forward_pre_hooks[key] = hook
        return RemovableHandle(self._forward_pre_hooks, key)

    def register_forward_hook(self, hook: _ForwardHook) -> RemovableHandle:
        """Register a hook called after forward(). Signature: hook(module, args, output) -> output | None."""
        key = _next_hook_id()
        self._forward_hooks[key] = hook
        return RemovableHandle(self._forward_hooks, key)

    def register_full_backward_hook(self, hook: _BackwardHook) -> RemovableHandle:
        """Register a backward hook. Returns a RemovableHandle."""
        key = _next_hook_id()
        self._backward_hooks[key] = hook
        return RemovableHandle(self._backward_hooks, key)

    def register_backward_hook(self, hook: _BackwardHook) -> RemovableHandle:
        """Deprecated alias for register_full_backward_hook."""
        return self.register_full_backward_hook(hook)

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        extra = self.extra_repr()
        cls_name = type(self).__name__
        if not self._modules:
            return f"{cls_name}({extra})" if extra else f"{cls_name}()"
        lines = [f"{cls_name}("]
        if extra:
            lines.append(f"  {extra}")
        for name, m in self._modules.items():
            mod_repr = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_repr}")
        lines.append(")")
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """Override to add extra repr info (e.g. Linear shows in_features, etc.)."""
        return ""
