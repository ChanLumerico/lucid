"""
state_dict save/load helpers for Module.
"""

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._types import StateDict


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
) -> tuple[list[str], list[str]]:
    """Load parameters from state_dict. Returns (missing_keys, unexpected_keys)."""
    own_state = module.state_dict(keep_vars=True)
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    for key in own_state:
        if key not in state_dict:
            missing_keys.append(key)

    for key in state_dict:
        if key not in own_state:
            unexpected_keys.append(key)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Missing keys: {missing_keys}; Unexpected keys: {unexpected_keys}"
        )

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

        if attr_name in sub._parameters:
            old_p = sub._parameters[attr_name]
            if old_p is not None:
                # Clone src storage onto the parameter's device, preserving requires_grad.
                new_impl = _C_engine.contiguous(src._impl).clone_with_grad(
                    old_p.requires_grad
                )
                old_p._impl = new_impl
        elif attr_name in sub._buffers:
            old_b = sub._buffers[attr_name]
            new_impl = _C_engine.contiguous(src._impl).clone_with_grad(False)
            sub._buffers[attr_name] = _wrap(new_impl)

    return missing_keys, unexpected_keys
