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
    r"""Reparametrise a weight as direction times magnitude (Salimans & Kingma 2016).

    Replaces ``module.<name>`` with the derived tensor

    .. math::

        \mathbf{W} \;=\; g \,\cdot\, \frac{\mathbf{v}}{\|\mathbf{v}\|}

    where ``g`` is a learnable scale (one entry per slice along ``dim``)
    and ``v`` is a learnable direction sharing the original weight's
    shape.  Decoupling magnitude from direction often accelerates
    convergence and improves conditioning, especially in deep
    convolutional and recurrent networks.

    Parameters
    ----------
    module : Module
        The module whose parameter is to be reparametrised.  Mutated in
        place — the original leaf parameter is removed and replaced with
        a derived tensor recomputed before every forward call.
    name : str, optional
        Attribute name of the parameter to normalise.  Default ``"weight"``.
    dim : int, optional
        Axis along which the scale ``g`` has its own entry; norms are
        reduced over every other axis.  ``0`` (the default) is correct
        for ``Linear`` and ``Conv*`` (one scale per output channel).
        Use ``-1`` for output-row Linear layouts.

    Returns
    -------
    Module
        The same module, now carrying ``<name>_g`` and ``<name>_v`` as
        leaf parameters in place of the original ``<name>``.

    Notes
    -----
    Two new parameters are registered:

    * ``<name>_g`` — broadcast-shaped tensor (``1`` along every axis
      except ``dim``) holding the per-slice magnitudes.
    * ``<name>_v`` — same shape as the original weight, holding the
      unnormalised direction.

    A forward pre-hook recomputes :math:`\mathbf{W} = g \cdot \mathbf{v}/\|\mathbf{v}\|`
    so subsequent autograd / inference sees the same shape as before.
    Invert with :func:`remove_weight_norm`.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import weight_norm
    >>> layer = weight_norm(nn.Linear(128, 64))
    >>> layer.weight_g.shape, layer.weight_v.shape
    ((64, 1), (64, 128))
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
    r"""Reverse :func:`weight_norm` and restore a plain leaf parameter.

    Materialises :math:`\mathbf{W} = g \cdot \mathbf{v}/\|\mathbf{v}\|`
    one last time, writes the result back as a single leaf parameter
    ``module.<name>``, drops ``<name>_g`` / ``<name>_v``, and detaches
    the forward pre-hook.  Typical use is right before exporting a model
    for inference where the reparametrised form is not needed.

    Parameters
    ----------
    module : Module
        Module previously passed through :func:`weight_norm`.
    name : str, optional
        Attribute name of the parameter to un-normalise.  Must match the
        ``name`` used at registration.  Default ``"weight"``.

    Returns
    -------
    Module
        The same module, with ``<name>`` restored as a plain
        :class:`~lucid.nn.parameter.Parameter` and the ``g``/``v``
        helpers removed.

    Raises
    ------
    ValueError
        If no weight-norm registration exists on ``<name>``.

    Notes
    -----
    The materialised parameter is detached from the autograd graph — any
    history accumulated through the reparametrisation is discarded.  If
    you need the original ``v`` direction back, copy it out *before*
    calling this.

    Examples
    --------
    >>> remove_weight_norm(layer)
    >>> "weight_g" in dict(layer.named_parameters())
    False
    """
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
