"""Weight Normalization (Salimans & Kingma, 2016).

Replaces a module's ``weight`` parameter with the reparametrisation

    weight = g * v / ||v||_dim

where ``g`` is a learnable scale (one entry per slice along ``dim``) and ``v``
is a learnable direction with the same shape as the original weight. The
reparametrised weight is recomputed before each forward call via a pre-hook.

Mirrors ``reference framework.nn.utils.weight_norm`` (the legacy non-parametrize variant).
"""

from typing import Any

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter

# Sentinel attribute on a Module to record which parameter name carries an
# active weight-norm registration; used by ``remove_weight_norm`` to undo it.
_WN_HOOK_ATTR: str = "_weight_norm_hooks"


def _norm_along_dim(v: Tensor, dim: int) -> Tensor:
    """Compute ||v|| reduced over every axis except ``dim``."""
    ndim: int = len(v.shape)
    if dim < 0:
        dim += ndim
    keep_axes: list[int] = [i for i in range(ndim) if i != dim]
    if not keep_axes:
        # 1-D weight — the norm is the full L2 norm scalar.
        return lucid.sqrt((v * v).sum())
    return lucid.sqrt((v * v).sum(dim=keep_axes, keepdim=True))


def _compute_weight(g: Tensor, v: Tensor, dim: int) -> Tensor:
    norm: Tensor = _norm_along_dim(v, dim)
    return g * (v / norm)


def weight_norm(module: Module, name: str = "weight", dim: int = 0) -> Module:
    """Apply weight normalisation to ``module``'s ``name`` parameter in-place.

    After the call, ``module.{name}`` is no longer a leaf parameter — it's
    replaced with ``g * v / ||v||_dim`` recomputed from the new
    ``module.{name}_g`` and ``module.{name}_v`` parameters before each forward.
    """
    if not isinstance(module, Module):
        raise TypeError(f"weight_norm requires a Module, got {type(module).__name__}")
    if not hasattr(module, name):
        raise AttributeError(f"module has no parameter '{name}'")
    weight: Parameter = getattr(module, name)
    if not isinstance(weight, Parameter):
        raise TypeError(
            f"'{name}' must be a Parameter to apply weight_norm, "
            f"got {type(weight).__name__}"
        )

    ndim: int = len(weight.shape)
    if dim < 0:
        dim += ndim

    # g shape: same as weight, but every axis except ``dim`` is 1 (so it
    # broadcasts correctly during the multiply).
    g_shape: list[int] = [1] * ndim
    if ndim > 0:
        g_shape[dim] = int(weight.shape[dim])
    initial_norm: Tensor = _norm_along_dim(weight.detach(), dim)
    g_param: Parameter = Parameter(initial_norm.reshape(g_shape))
    v_param: Parameter = Parameter(weight.detach())

    # Drop the old leaf parameter and install ``g`` / ``v`` next to it.
    del module._parameters[name]
    module.register_parameter(name + "_g", g_param)
    module.register_parameter(name + "_v", v_param)

    def _pre_hook(mod: Module, inputs: Any) -> None:  # noqa: ANN401
        # Recompute weight from the current g/v before every forward.
        g: Parameter = getattr(mod, name + "_g")
        v: Parameter = getattr(mod, name + "_v")
        object.__setattr__(mod, name, _compute_weight(g, v, dim))

    handle = module.register_forward_pre_hook(_pre_hook)  # type: ignore[arg-type]

    # Track the registration so remove_weight_norm can find it.
    hooks: dict[str, Any] = getattr(module, _WN_HOOK_ATTR, {})
    hooks[name] = (handle, dim)
    object.__setattr__(module, _WN_HOOK_ATTR, hooks)

    # Trigger the hook once so the attribute exists immediately (without
    # waiting for a forward call).
    object.__setattr__(module, name, _compute_weight(g_param, v_param, dim))
    return module


def remove_weight_norm(module: Module, name: str = "weight") -> Module:
    """Reverse ``weight_norm``: collapse ``g``/``v`` back into a single
    leaf parameter ``name`` and detach the pre-hook."""
    hooks: dict[str, Any] = getattr(module, _WN_HOOK_ATTR, {})
    if name not in hooks:
        raise ValueError(f"weight_norm not registered on '{name}'")
    handle, dim = hooks.pop(name)
    handle.remove()
    g: Parameter = getattr(module, name + "_g")
    v: Parameter = getattr(module, name + "_v")
    materialised: Tensor = _compute_weight(g, v, dim).detach()
    del module._parameters[name + "_g"]
    del module._parameters[name + "_v"]
    # Drop the cached non-leaf attribute and re-install as a fresh Parameter.
    if name in module.__dict__:
        del module.__dict__[name]
    module.register_parameter(name, Parameter(materialised))
    if not hooks:
        try:
            object.__delattr__(module, _WN_HOOK_ATTR)
        except AttributeError:
            pass
    return module
