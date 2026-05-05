"""
nn.Module: base class for all neural network layers.
"""

from collections import OrderedDict
from typing import Callable, ClassVar, Iterator, Self

from lucid._C import engine as _C_engine
from lucid._tensor.tensor import Tensor
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.parameter import Parameter
from lucid.nn.hooks import (
    _GLOBAL_BACKWARD_HOOKS,
    _GLOBAL_BACKWARD_PRE_HOOKS,
    _GLOBAL_FORWARD_HOOKS,
    _GLOBAL_FORWARD_PRE_HOOKS,
    RemovableHandle,
)
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
        object.__setattr__(self, "_non_persistent_buffers", set())
        object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_forward_hooks", OrderedDict())
        object.__setattr__(self, "_forward_pre_hooks_with_kwargs", set())
        object.__setattr__(self, "_forward_hooks_with_kwargs", set())
        object.__setattr__(self, "_forward_hooks_always_called", set())
        object.__setattr__(self, "_backward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_backward_hooks", OrderedDict())
        object.__setattr__(self, "training", True)

    def forward(self, *args: Tensor, **kwargs: object) -> _ModuleOutput:
        """Override in subclasses to define the computation."""
        raise NotImplementedError(f"{type(self).__name__}.forward() not implemented")

    def __call__(self, *args: Tensor, **kwargs: object) -> _ModuleOutput:
        for hook, with_kwargs in _GLOBAL_FORWARD_PRE_HOOKS.values():
            args, kwargs = self._call_forward_pre_hook(
                hook, args, kwargs, with_kwargs=with_kwargs
            )
        for key, hook in self._forward_pre_hooks.items():
            args, kwargs = self._call_forward_pre_hook(
                hook,
                args,
                kwargs,
                with_kwargs=key in self._forward_pre_hooks_with_kwargs,
            )

        backward_state = None
        if self._has_backward_hooks():
            args, backward_state = self._prepare_backward_hooks(args)

        output: _ModuleOutput | None = None
        try:
            output = self.forward(*args, **kwargs)
        except Exception:
            self._call_always_forward_hooks(args, kwargs, output)
            raise

        for hook, with_kwargs, _ in _GLOBAL_FORWARD_HOOKS.values():
            output = self._call_forward_hook(
                hook, args, kwargs, output, with_kwargs=with_kwargs
            )
        for key, hook in self._forward_hooks.items():
            output = self._call_forward_hook(
                hook,
                args,
                kwargs,
                output,
                with_kwargs=key in self._forward_hooks_with_kwargs,
            )
        if backward_state is not None:
            output = self._attach_output_backward_hooks(output, backward_state)
        return output

    def _call_forward_pre_hook(
        self,
        hook: Callable[..., object],
        args: tuple[Tensor, ...],
        kwargs: dict[str, object],
        *,
        with_kwargs: bool,
    ) -> tuple[tuple[Tensor, ...], dict[str, object]]:
        if with_kwargs:
            result = hook(self, args, kwargs)
            if result is None:
                return args, kwargs
            if (
                not isinstance(result, tuple)
                or len(result) != 2
                or not isinstance(result[1], dict)
            ):
                raise RuntimeError(
                    "forward pre-hook with kwargs must return None or (args, kwargs)"
                )
            new_args = result[0]
            if not isinstance(new_args, tuple):
                new_args = (new_args,)
            return new_args, result[1]

        result = hook(self, args)
        if result is None:
            return args, kwargs
        if not isinstance(result, tuple):
            result = (result,)
        return result, kwargs

    def _call_forward_hook(
        self,
        hook: Callable[..., object],
        args: tuple[Tensor, ...],
        kwargs: dict[str, object],
        output: _ModuleOutput,
        *,
        with_kwargs: bool,
    ) -> _ModuleOutput:
        if with_kwargs:
            hook_out = hook(self, args, kwargs, output)
        else:
            hook_out = hook(self, args, output)
        return output if hook_out is None else hook_out

    def _call_always_forward_hooks(
        self,
        args: tuple[Tensor, ...],
        kwargs: dict[str, object],
        output: _ModuleOutput | None,
    ) -> None:
        for hook, with_kwargs, always_call in _GLOBAL_FORWARD_HOOKS.values():
            if not always_call:
                continue
            if with_kwargs:
                hook(self, args, kwargs, output)
            else:
                hook(self, args, output)
        for key, hook in self._forward_hooks.items():
            if key not in self._forward_hooks_always_called:
                continue
            if key in self._forward_hooks_with_kwargs:
                hook(self, args, kwargs, output)
            else:
                hook(self, args, output)

    def _has_backward_hooks(self) -> bool:
        return (
            bool(_GLOBAL_BACKWARD_PRE_HOOKS)
            or bool(_GLOBAL_BACKWARD_HOOKS)
            or bool(self._backward_pre_hooks)
            or bool(self._backward_hooks)
        )

    def _prepare_backward_hooks(
        self,
        args: tuple[Tensor, ...],
    ) -> tuple[tuple[Tensor, ...], _ModuleBackwardState]:
        state = _ModuleBackwardState(self, len(args))
        if hasattr(_C_engine, "_create_module_backward_hook_state"):
            state.cpp_state = _C_engine._create_module_backward_hook_state(
                len(args),
                state.apply_backward_pre_hooks_from_cpp,
                state.apply_full_backward_hooks_from_cpp,
            )
            entries: list[tuple[int, _C_engine.TensorImpl]] = []
            for idx, arg in enumerate(args):
                if isinstance(arg, Tensor) and arg.requires_grad:
                    entries.append((idx, _unwrap(arg)))
                    state.input_tensor_indices.append(idx)
            if not entries:
                return args, state
            wrapped_impls = _C_engine._wrap_module_backward_inputs(
                state.cpp_state, entries
            )
            wrapped_args = list(args)
            for (idx, _), impl in zip(entries, wrapped_impls, strict=True):
                wrapped_args[idx] = _wrap(impl)
            return tuple(wrapped_args), state

        wrapped: list[Tensor] = []
        for idx, arg in enumerate(args):
            if isinstance(arg, Tensor) and arg.requires_grad:
                wrapped.append(_ModuleInputBackwardHookFunction.apply(arg, state, idx))
                state.input_tensor_indices.append(idx)
            else:
                wrapped.append(arg)
        return tuple(wrapped), state

    def _attach_output_backward_hooks(
        self,
        output: _ModuleOutput,
        state: _ModuleBackwardState,
    ) -> _ModuleOutput:
        if state.cpp_state is not None:
            return self._attach_cpp_output_backward_hooks(output, state)

        if isinstance(output, tuple):
            state.n_outputs = sum(1 for item in output if isinstance(item, Tensor))
            output_idx = 0
            wrapped_output: list[Tensor] = []
            for item in output:
                if isinstance(item, Tensor):
                    wrapped_output.append(
                        _ModuleOutputBackwardHookFunction.apply(item, state, output_idx)
                    )
                    output_idx += 1
                else:
                    wrapped_output.append(item)
            return tuple(wrapped_output)
        if not isinstance(output, Tensor):
            return output
        state.n_outputs = 1
        return _ModuleOutputBackwardHookFunction.apply(output, state, 0)

    def _attach_cpp_output_backward_hooks(
        self,
        output: _ModuleOutput,
        state: _ModuleBackwardState,
    ) -> _ModuleOutput:
        if isinstance(output, tuple):
            n_outputs = sum(1 for item in output if isinstance(item, Tensor))
            state.n_outputs = n_outputs
            entries: list[tuple[int, _C_engine.TensorImpl]] = []
            output_idx = 0
            positions: list[int] = []
            for pos, item in enumerate(output):
                if isinstance(item, Tensor):
                    if item.requires_grad:
                        entries.append((output_idx, _unwrap(item)))
                        positions.append(pos)
                    output_idx += 1
            if not entries:
                return output
            wrapped_impls = _C_engine._wrap_module_backward_outputs(
                state.cpp_state, entries, n_outputs
            )
            wrapped_output = list(output)
            for pos, impl in zip(positions, wrapped_impls, strict=True):
                wrapped_output[pos] = _wrap(impl)
            return tuple(wrapped_output)

        if not isinstance(output, Tensor):
            return output
        state.n_outputs = 1
        if not output.requires_grad:
            return output
        wrapped_impl = _C_engine._wrap_module_backward_outputs(
            state.cpp_state, [(0, _unwrap(output))], 1
        )[0]
        return _wrap(wrapped_impl)

    def __setattr__(self, name: str, value: object) -> None:
        if not isinstance(name, str):
            raise TypeError("module attribute name must be a string")
        if "." in name:
            raise KeyError("module attribute name cannot contain '.'")
        if name == "":
            raise KeyError("module attribute name cannot be empty")

        # Remove from existing dicts before re-routing
        for d in (self._parameters, self._buffers, self._modules):
            if name in d:
                del d[name]
        if name in self._non_persistent_buffers:
            self._non_persistent_buffers.remove(name)

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

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
            return
        if name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
            return
        if name in self._modules:
            del self._modules[name]
            return
        object.__delattr__(self, name)

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
            once, even if referenced by multiple attributes. Mirrors reference framework.
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
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Tensor]]:
        """Yield (name, buffer) pairs."""
        seen: set[int] = set()
        for name, b in self._buffers.items():
            if b is not None:
                if remove_duplicate:
                    if id(b) in seen:
                        continue
                    seen.add(id(b))
                yield (f"{prefix}.{name}" if prefix else name), b
        if recurse:
            for mname, m in self._modules.items():
                subprefix = f"{prefix}.{mname}" if prefix else mname
                for full_name, b in m.named_buffers(
                    subprefix, recurse=True, remove_duplicate=False
                ):
                    if remove_duplicate:
                        if id(b) in seen:
                            continue
                        seen.add(id(b))
                    yield full_name, b

    def modules(self) -> Iterator[Module]:
        """Yield this module and all submodules (depth-first)."""
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: set[int] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Module]]:
        """Yield (name, module) pairs."""
        if memo is None:
            memo = set()
        if remove_duplicate and id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, m in self._modules.items():
            if m is None:
                continue
            subprefix = f"{prefix}.{name}" if prefix else name
            yield from m.named_modules(
                memo=memo, prefix=subprefix, remove_duplicate=remove_duplicate
            )

    def children(self) -> Iterator[Module]:
        """Yield direct child modules."""
        for module in self._modules.values():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Yield (name, child_module) pairs."""
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

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
        self._validate_child_name(name, "parameter")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign '{type(param).__name__}' object as parameter '{name}'"
            )
        if name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        if name in self._modules:
            del self._modules[name]
        if param is None:
            self._parameters[name] = None  # type: ignore[assignment]
        else:
            self._parameters[name] = param

    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        """Register a buffer tensor. Non-persistent buffers are excluded from state_dict."""
        self._validate_child_name(name, "buffer")
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(
                f"cannot assign '{type(tensor).__name__}' object as buffer '{name}'"
            )
        if name in self._parameters:
            del self._parameters[name]
        if name in self._modules:
            del self._modules[name]
        if not persistent:
            self._non_persistent_buffers.add(name)
        else:
            self._non_persistent_buffers.discard(name)
        self._buffers[name] = tensor

    def add_module(self, name: str, module: Module | None) -> None:
        """Add a child module."""
        self._validate_child_name(name, "module")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        if module is not None and not isinstance(module, Module):
            raise TypeError(
                f"cannot assign '{type(module).__name__}' object as child module '{name}'"
            )
        if name in self._parameters:
            del self._parameters[name]
        if name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers.discard(name)
        if module is None:
            self._modules[name] = None  # type: ignore[assignment]
        else:
            self._modules[name] = module

    def register_module(self, name: str, module: Module | None) -> None:
        """Alias for add_module."""
        self.add_module(name, module)

    def _validate_child_name(self, name: str, kind: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"{kind} name must be a string")
        if name == "":
            raise KeyError(f"{kind} name cannot be empty")
        if "." in name:
            raise KeyError(f"{kind} name cannot contain '.'")

    # ── training mode ────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> Self:
        """Set this module and all children to training mode."""
        if not isinstance(mode, bool):
            raise TypeError(f"train() requires a bool, got {type(mode).__name__}")
        self.training = mode
        for m in self._modules.values():
            if m is not None:
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
    ) -> object:
        """Load parameters from a state_dict."""
        from lucid.nn._state_dict import _load_from_state_dict

        return _load_from_state_dict(self, state_dict, strict=strict)

    # ── hooks ─────────────────────────────────────────────────────────────

    def register_forward_pre_hook(
        self,
        hook: _ForwardPreHook,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        """Register a hook called before forward()."""
        key = _next_hook_id()
        self._forward_pre_hooks[key] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs.add(key)
        if prepend:
            self._forward_pre_hooks.move_to_end(key, last=False)
        return RemovableHandle(
            self._forward_pre_hooks, key, (self._forward_pre_hooks_with_kwargs,)
        )

    def register_forward_hook(
        self,
        hook: _ForwardHook,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        """Register a hook called after forward()."""
        key = _next_hook_id()
        self._forward_hooks[key] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs.add(key)
        if always_call:
            self._forward_hooks_always_called.add(key)
        if prepend:
            self._forward_hooks.move_to_end(key, last=False)
        return RemovableHandle(
            self._forward_hooks,
            key,
            (self._forward_hooks_with_kwargs, self._forward_hooks_always_called),
        )

    def register_full_backward_pre_hook(
        self,
        hook: _BackwardHook,
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        """Register a hook to be called before backward hooks."""
        key = _next_hook_id()
        self._backward_pre_hooks[key] = hook
        if prepend:
            self._backward_pre_hooks.move_to_end(key, last=False)
        return RemovableHandle(self._backward_pre_hooks, key)

    def register_full_backward_hook(
        self,
        hook: _BackwardHook,
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        """Register a backward hook. Returns a RemovableHandle."""
        key = _next_hook_id()
        self._backward_hooks[key] = hook
        if prepend:
            self._backward_hooks.move_to_end(key, last=False)
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


class _ModuleBackwardState:
    def __init__(self, module: Module, n_inputs: int) -> None:
        self.module = module
        self.n_inputs = n_inputs
        self.n_outputs = 0
        self.input_tensor_indices: list[int] = []
        self.grad_inputs: list[Tensor | None] = [None] * n_inputs
        self.grad_outputs: list[Tensor | None] = []
        self.pre_hooks_ran = False
        self.full_hooks_ran = False
        self.cpp_state: object | None = None

    def set_num_outputs(self, n_outputs: int) -> None:
        if not self.grad_outputs:
            self.grad_outputs = [None] * n_outputs

    def apply_backward_pre_hooks(self, index: int, grad_output: Tensor) -> Tensor:
        self.set_num_outputs(max(self.n_outputs, index + 1))
        self.grad_outputs[index] = grad_output

        # For a single Tensor output, pre-hooks can transform the actual
        # gradient flowing into the module. For multiple outputs, each output
        # edge arrives separately; apply the hook to the currently available
        # slot while preserving other slots as None.
        grad_outputs = tuple(self.grad_outputs)
        for hook in _GLOBAL_BACKWARD_PRE_HOOKS.values():
            result = hook(self.module, grad_outputs)
            if result is not None:
                grad_outputs = result if isinstance(result, tuple) else (result,)
        for hook in self.module._backward_pre_hooks.values():
            result = hook(self.module, grad_outputs)
            if result is not None:
                grad_outputs = result if isinstance(result, tuple) else (result,)

        self.pre_hooks_ran = True
        updated = grad_outputs[index] if index < len(grad_outputs) else None
        return updated if isinstance(updated, Tensor) else grad_output

    def apply_backward_pre_hooks_from_cpp(
        self, grad_output_impls: tuple[_C_engine.TensorImpl | None, ...]
    ) -> tuple[_C_engine.TensorImpl | None, ...] | None:
        self.grad_outputs = [
            _wrap(g) if isinstance(g, _C_engine.TensorImpl) else None
            for g in grad_output_impls
        ]
        grad_outputs = tuple(self.grad_outputs)
        for hook in _GLOBAL_BACKWARD_PRE_HOOKS.values():
            result = hook(self.module, grad_outputs)
            if result is not None:
                grad_outputs = result if isinstance(result, tuple) else (result,)
        for hook in self.module._backward_pre_hooks.values():
            result = hook(self.module, grad_outputs)
            if result is not None:
                grad_outputs = result if isinstance(result, tuple) else (result,)
        self.pre_hooks_ran = True
        self.grad_outputs = [
            item if isinstance(item, Tensor) else None for item in grad_outputs
        ]
        return tuple(
            _unwrap(item) if isinstance(item, Tensor) else None for item in grad_outputs
        )

    def apply_full_backward_hooks_for_input(
        self,
        index: int,
        grad_input: Tensor,
    ) -> Tensor:
        self.grad_inputs[index] = grad_input
        grad_inputs = tuple(self.grad_inputs)
        grad_outputs = tuple(self.grad_outputs)

        for hook in _GLOBAL_BACKWARD_HOOKS.values():
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                grad_inputs = result if isinstance(result, tuple) else (result,)
        for hook in self.module._backward_hooks.values():
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                grad_inputs = result if isinstance(result, tuple) else (result,)

        self.full_hooks_ran = True
        updated = grad_inputs[index] if index < len(grad_inputs) else None
        return updated if isinstance(updated, Tensor) else grad_input

    def apply_full_backward_hooks_from_cpp(
        self,
        grad_input_impls: tuple[_C_engine.TensorImpl | None, ...],
        grad_output_impls: tuple[_C_engine.TensorImpl | None, ...],
    ) -> tuple[_C_engine.TensorImpl | None, ...] | None:
        grad_inputs = tuple(
            _wrap(g) if isinstance(g, _C_engine.TensorImpl) else None
            for g in grad_input_impls
        )
        grad_outputs = tuple(
            _wrap(g) if isinstance(g, _C_engine.TensorImpl) else None
            for g in grad_output_impls
        )
        for hook in _GLOBAL_BACKWARD_HOOKS.values():
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                grad_inputs = result if isinstance(result, tuple) else (result,)
        for hook in self.module._backward_hooks.values():
            result = hook(self.module, grad_inputs, grad_outputs)
            if result is not None:
                grad_inputs = result if isinstance(result, tuple) else (result,)
        self.full_hooks_ran = True
        self.grad_inputs = [
            item if isinstance(item, Tensor) else None for item in grad_inputs
        ]
        self.grad_outputs = [
            item if isinstance(item, Tensor) else None for item in grad_outputs
        ]
        return tuple(
            _unwrap(item) if isinstance(item, Tensor) else None for item in grad_inputs
        )

    def apply_full_backward_hooks_without_inputs(self) -> None:
        if self.full_hooks_ran or self.input_tensor_indices:
            return
        grad_inputs: tuple[Tensor | None, ...] = ()
        grad_outputs = tuple(self.grad_outputs)
        for hook in _GLOBAL_BACKWARD_HOOKS.values():
            hook(self.module, grad_inputs, grad_outputs)
        for hook in self.module._backward_hooks.values():
            hook(self.module, grad_inputs, grad_outputs)
        self.full_hooks_ran = True


class _ModuleInputBackwardHookFunction:
    @staticmethod
    def apply(x: Tensor, state: _ModuleBackwardState, index: int) -> Tensor:
        from lucid.autograd import Function

        class _InputHook(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.state = state
                ctx.index = index
                return x

            @staticmethod
            def backward(ctx, grad_input):
                return ctx.state.apply_full_backward_hooks_for_input(
                    ctx.index, grad_input
                )

        return _InputHook.apply(x)


class _ModuleOutputBackwardHookFunction:
    @staticmethod
    def apply(output: Tensor, state: _ModuleBackwardState, index: int) -> Tensor:
        from lucid.autograd import Function

        class _OutputHook(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.state = state
                ctx.index = index
                return x

            @staticmethod
            def backward(ctx, grad_output):
                updated = ctx.state.apply_backward_pre_hooks(ctx.index, grad_output)
                ctx.state.apply_full_backward_hooks_without_inputs()
                return updated

        return _OutputHook.apply(output)
