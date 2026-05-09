"""state_dict save/load infrastructure for Module.

The protocol mirrors the reference framework:

* ``Module._save_to_state_dict`` writes own params / buffers into the dest dict.
  The top-level ``state_dict()`` walker recurses into children.
* ``Module._load_from_state_dict`` consumes its own keys from the supplied
  state_dict, mutating ``missing_keys``, ``unexpected_keys`` and
  ``error_msgs`` in place.  The top-level ``load_state_dict()`` driver
  recurses, runs pre/post hooks, and raises a single ``RuntimeError`` at
  the end when ``strict=True``.
* ``state_dict()`` attaches an ``_metadata`` attribute to the returned
  OrderedDict mapping ``module_path → {"version": <int>}`` for every module
  that defines a ``_version`` class attribute.  This metadata is forwarded
  to each child's ``_load_from_state_dict`` via the ``local_metadata`` arg
  for migration hooks to inspect.
"""

from collections import OrderedDict, namedtuple

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.nn.hooks import (
    _GLOBAL_LOAD_STATE_DICT_POST_HOOKS,
    _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS,
)
from lucid.nn.module import Module
from lucid._types import StateDict


class IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    """Return value of ``Module.load_state_dict``."""

    __slots__ = ()


# ── state_dict (save side) ────────────────────────────────────────────────────


def _build_metadata(module: Module, prefix: str = "") -> dict[str, dict[str, object]]:
    """Walk the module tree and collect per-module ``_version`` tags.

    Every module now carries at least ``{"version": 1}`` (base class default),
    matching the reference framework.  Subclasses may set ``_version = N``
    (N ≥ 2) to signal that their ``_load_from_state_dict`` performs key
    migration for older checkpoints.
    """
    metadata: dict[str, dict[str, object]] = {}
    version: int | None = getattr(type(module), "_version", None)
    key: str = prefix.rstrip(".") if prefix else ""
    # Always record when version is a non-None int (includes the base default 1).
    if isinstance(version, int):
        metadata[key] = {"version": version}
    for name, child in module._modules.items():
        if child is None:
            continue
        metadata.update(_build_metadata(child, f"{prefix}{name}."))
    return metadata


def _save_to_state_dict(
    module: Module,
    destination: dict[str, Tensor],
    prefix: str = "",
    keep_vars: bool = False,
) -> None:
    """Recursive walker — let each module write its own state, then recurse.

    A module may set the class attribute ``_state_dict_skip_recursion = True``
    to indicate that its ``_save_to_state_dict`` already serialised every
    descendant it cares about (e.g. by flattening a child sub-tree into a
    custom key namespace).  When set, the walker stops descending into
    that module's children.
    """
    module._save_to_state_dict(destination, prefix, keep_vars)
    if getattr(module, "_state_dict_skip_recursion", False):
        return
    for mname, child in module._modules.items():
        if child is None:
            continue
        _save_to_state_dict(child, destination, f"{prefix}{mname}.", keep_vars)


# ── state_dict (load side) ────────────────────────────────────────────────────


def _default_load_from_state_dict(
    module: Module,
    state_dict: StateDict,
    prefix: str,
    local_metadata: dict[str, object],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
    assign: bool = False,
) -> None:
    """Default ``_load_from_state_dict`` impl — params + persistent buffers.

    Subclasses should call this from their override (after any pre-processing)
    to share the standard copy / dtype-convert / shape-check logic.

    Parameters
    ----------
    assign : bool
        When ``True`` the loaded tensor is directly assigned as the new
        parameter / buffer object (no shape check, no dtype coercion).
        When ``False`` (default) data is copied into the existing tensor
        preserving its dtype and device.
    """
    from lucid.nn.parameter import Parameter

    persistent_buffers: dict[str, Tensor] = {
        name: b
        for name, b in module._buffers.items()
        if b is not None and name not in module._non_persistent_buffers
    }
    local_state: dict[str, Tensor] = {
        **{name: p for name, p in module._parameters.items() if p is not None},
        **persistent_buffers,
    }

    for name, attr in local_state.items():
        key: str = f"{prefix}{name}"
        if key not in state_dict:
            missing_keys.append(key)
            continue
        src: Tensor = state_dict[key]
        if not isinstance(src, Tensor):
            error_msgs.append(
                f"While copying parameter '{key}', expected Tensor, "
                f"got {type(src).__name__}"
            )
            continue

        if assign:
            # assign=True: replace the parameter/buffer object directly,
            # allowing shape and dtype changes (matches reference framework).
            needs_grad: bool = getattr(attr, "requires_grad", False)
            new_impl: object = _C_engine.contiguous(src._impl).clone_with_grad(needs_grad)
            if name in module._parameters:
                # Preserve Parameter wrapping.
                new_param = Parameter(_wrap(new_impl), requires_grad=needs_grad)
                module._parameters[name] = new_param
            else:
                module._buffers[name] = _wrap(new_impl)
        else:
            if tuple(src.shape) != tuple(attr.shape):
                error_msgs.append(
                    f"size mismatch for {key}: "
                    f"expected {tuple(attr.shape)}, got {tuple(src.shape)}"
                )
                continue
            # Copy with original dtype/device preserved.
            converted: Tensor = src.to(device=attr.device, dtype=attr.dtype)
            new_impl = _C_engine.contiguous(converted._impl).clone_with_grad(
                getattr(attr, "requires_grad", False)
            )
            if name in module._parameters:
                module._parameters[name]._impl = new_impl  # type: ignore[union-attr]
            else:
                module._buffers[name] = _wrap(new_impl)

    # extra_state (rare hook for opaque per-module state).
    extra_key: str = f"{prefix}_extra_state"
    if extra_key in state_dict:
        module.set_extra_state(state_dict[extra_key])
    elif module.get_extra_state() is not None:
        # Module advertises extra_state but checkpoint lacks it.
        missing_keys.append(extra_key)


def _enumerate_local_keys(module: Module, prefix: str) -> list[str]:
    """Flat list of keys this module owns at ``prefix`` (params + persistent buffers).

    Modules that flatten a child sub-tree into a custom key namespace
    can override the result by exposing a ``_local_state_dict_keys(prefix)``
    method.  When present, that method's return value replaces the default
    enumeration entirely.
    """
    custom = getattr(module, "_local_state_dict_keys", None)
    if callable(custom):
        return list(custom(prefix))
    keys: list[str] = []
    for name, p in module._parameters.items():
        if p is not None:
            keys.append(f"{prefix}{name}")
    for name, b in module._buffers.items():
        if b is not None and name not in module._non_persistent_buffers:
            keys.append(f"{prefix}{name}")
    if module.get_extra_state() is not None:
        keys.append(f"{prefix}_extra_state")
    return keys


def _walk_load(
    module: Module,
    state_dict: StateDict,
    prefix: str,
    metadata: dict[str, dict[str, object]],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
    post_hook_modules: list[Module],
    assign: bool = False,
) -> None:
    """Pre-order walk: handle this module, then recurse into children.

    Post-hooks are deferred — modules with registered post-hooks are
    appended to `post_hook_modules` and fired by the top-level driver
    after the full walk completes, so each hook sees the final
    IncompatibleKeys rather than a per-module snapshot.
    """
    local_meta: dict[str, object] = (
        metadata.get(prefix.rstrip("."), {}) if metadata else {}
    )

    # Pre-hooks (global → instance).
    for hook in _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS.values():
        hook(
            module,
            state_dict,
            prefix,
            local_meta,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    for hook in module._load_state_dict_pre_hooks.values():
        hook(
            module,
            state_dict,
            prefix,
            local_meta,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    if assign:
        # assign=True: use the internal default loader directly with assign
        # semantics (replaces parameter objects rather than copying data).
        # Custom _load_from_state_dict overrides are bypassed intentionally —
        # the assign path is inherently low-level.
        _default_load_from_state_dict(
            module,
            state_dict,
            prefix,
            local_meta,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            assign=True,
        )
    else:
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_meta,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    if module._load_state_dict_post_hooks:
        post_hook_modules.append(module)

    if getattr(module, "_state_dict_skip_recursion", False):
        return

    for name, child in module._modules.items():
        if child is None:
            continue
        _walk_load(
            child,
            state_dict,
            f"{prefix}{name}.",
            metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            post_hook_modules,
            assign,
        )


def load_state_dict(
    module: Module,
    state_dict: StateDict,
    strict: bool = True,
    assign: bool = False,
) -> IncompatibleKeys:
    """Driver for ``Module.load_state_dict``.

    Pre-order recursion: at each module, run pre-hooks, then
    ``_load_from_state_dict``, then post-hooks, then recurse into children.

    ``state_dict`` may carry an ``_metadata`` attribute (set by
    ``Module.state_dict()``) that maps module paths to per-module metadata
    such as ``{"version": N}``.  Each module's hook receives the relevant
    slice as ``local_metadata``.
    """
    metadata: dict[str, dict[str, object]] = (
        getattr(state_dict, "_metadata", None) or {}
    )
    # We must operate on a mutable copy so hooks may rename keys safely.
    state_dict = OrderedDict(state_dict)
    state_dict._metadata = metadata  # type: ignore[attr-defined]

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    error_msgs: list[str] = []
    post_hook_modules: list[Module] = []

    _walk_load(
        module,
        state_dict,
        "",
        metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        post_hook_modules,
        assign,
    )

    # Compute final unexpected_keys based on the post-hook state_dict
    # (pre-hooks may have renamed/added/removed keys).  Anything still
    # present that no module claimed is "unexpected".
    expected_keys: set[str] = set()
    _collect_expected(module, "", expected_keys)
    unexpected_keys = [k for k in state_dict.keys() if k not in expected_keys]

    incompatible: IncompatibleKeys = IncompatibleKeys(missing_keys, unexpected_keys)

    # Fire post-hooks (instance + global) with the final IncompatibleKeys.
    for m in post_hook_modules:
        for hook in m._load_state_dict_post_hooks.values():
            hook(m, incompatible)
    if _GLOBAL_LOAD_STATE_DICT_POST_HOOKS:

        def _walk_post(mod: Module) -> None:
            for hook in _GLOBAL_LOAD_STATE_DICT_POST_HOOKS.values():
                hook(mod, incompatible)
            for child in mod._modules.values():
                if child is not None:
                    _walk_post(child)

        _walk_post(module)

    # Strict mode: turn missing/unexpected into errors.
    if strict:
        if missing_keys:
            error_msgs.insert(
                0,
                f"Missing key(s) in state_dict: {', '.join(repr(k) for k in missing_keys)}.",
            )
        if unexpected_keys:
            error_msgs.insert(
                0,
                f"Unexpected key(s) in state_dict: {', '.join(repr(k) for k in unexpected_keys)}.",
            )

    if error_msgs:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                type(module).__name__, "\n\t".join(error_msgs)
            )
        )

    return incompatible


def _collect_expected(module: Module, prefix: str, out: set[str]) -> None:
    out.update(_enumerate_local_keys(module, prefix))
    if getattr(module, "_state_dict_skip_recursion", False):
        return
    for name, child in module._modules.items():
        if child is None:
            continue
        _collect_expected(child, f"{prefix}{name}.", out)
