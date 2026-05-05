"""state_dict save/load helpers for Module."""

from collections import namedtuple

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._types import StateDict


class IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    """Return value for load_state_dict."""

    __slots__ = ()


def _save_to_state_dict(
    module: Module,
    prefix: str = "",
    keep_vars: bool = False,
) -> StateDict:
    """Recursively collect parameters and persistent buffers into a flat dict."""
    result: StateDict = {}
    non_persistent: set[str] = getattr(module, "_non_persistent_buffers", set())

    for name, p in module._parameters.items():
        if p is not None:
            key = f"{prefix}{name}"
            result[key] = p if keep_vars else p.detach()

    for name, b in module._buffers.items():
        if b is not None and name not in non_persistent:
            key = f"{prefix}{name}"
            result[key] = b if keep_vars else b.detach()

    for mname, m in module._modules.items():
        if m is None:
            continue
        subprefix = f"{prefix}{mname}."
        result.update(_save_to_state_dict(m, prefix=subprefix, keep_vars=keep_vars))

    # Allow modules to include extra state
    extra = module.get_extra_state()
    if extra is not None:
        result[f"{prefix}_extra_state"] = extra

    return result


def _load_from_state_dict(
    module: Module,
    state_dict: StateDict,
    strict: bool = True,
) -> IncompatibleKeys:
    """Load parameters from state_dict."""
    _materialize_lazy_modules(module, state_dict)
    own_state = module.state_dict(keep_vars=True)
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    error_msgs: list[str] = []

    for key in own_state:
        if key not in state_dict:
            missing_keys.append(key)

    for key in state_dict:
        if key not in own_state:
            unexpected_keys.append(key)

    if strict and (missing_keys or unexpected_keys):
        if missing_keys:
            error_msgs.append(f"Missing key(s): {missing_keys}")
        if unexpected_keys:
            error_msgs.append(f"Unexpected key(s): {unexpected_keys}")

    for key, src in state_dict.items():
        if key not in own_state:
            continue
        # Navigate to the right sub-module
        parts = key.split(".")
        sub: Module = module
        for part in parts[:-1]:
            if part == "_extra_state":
                break
            sub = sub._modules[part]
        attr_name = parts[-1]

        if attr_name == "_extra_state":
            sub.set_extra_state(src)
            continue

        if not isinstance(src, Tensor):
            error_msgs.append(
                f"While copying '{key}': expected Tensor, got {type(src).__name__}"
            )
            continue

        if attr_name in sub._parameters:
            old_p = sub._parameters[attr_name]
            if old_p is not None:
                if tuple(src.shape) != tuple(old_p.shape):
                    error_msgs.append(
                        f"size mismatch for {key}: expected {old_p.shape}, got {src.shape}"
                    )
                    continue
                converted = src.to(device=old_p.device, dtype=old_p.dtype)
                new_impl = _C_engine.contiguous(converted._impl).clone_with_grad(
                    old_p.requires_grad
                )
                old_p._impl = new_impl
        elif attr_name in sub._buffers:
            old_b = sub._buffers[attr_name]
            if old_b is not None and tuple(src.shape) != tuple(old_b.shape):
                error_msgs.append(
                    f"size mismatch for {key}: expected {old_b.shape}, got {src.shape}"
                )
                continue
            converted = (
                src if old_b is None else src.to(device=old_b.device, dtype=old_b.dtype)
            )
            new_impl = _C_engine.contiguous(converted._impl).clone_with_grad(False)
            sub._buffers[attr_name] = _wrap(new_impl)

    if error_msgs:
        raise RuntimeError(
            "Error(s) in loading state_dict:\n\t" + "\n\t".join(error_msgs)
        )

    return IncompatibleKeys(missing_keys, unexpected_keys)


def _materialize_lazy_modules(
    module: Module,
    state_dict: StateDict,
    prefix: str = "",
) -> None:
    initializer = getattr(module, "_initialize_from_state_dict", None)
    if initializer is not None:
        initializer(state_dict, prefix)

    for name, child in module._modules.items():
        if child is None:
            continue
        _materialize_lazy_modules(child, state_dict, f"{prefix}{name}.")
