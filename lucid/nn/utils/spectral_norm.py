"""Spectral Normalization (Miyato et al., 2018).

Reparametrises a module's ``weight`` parameter as

    weight = W / σ(W)

where ``σ(W)`` is the largest singular value of ``W``, estimated cheaply by
running one or more power-iteration steps on a pair of buffers ``u`` and
``v``.  Constraining the spectral norm of the linear/conv weight bounds
the layer's Lipschitz constant, which stabilises GAN discriminators and
some Transformer variants.

Reproduction follows the reference framework's legacy ``spectral_norm``
API (the parametrize-based variant is a separate piece of work).  Each
forward call:

  1. Reads the current ``weight_orig`` parameter.
  2. Runs ``n_power_iterations`` updates of ``u`` and ``v``:
         v ← normalize(W^T u)
         u ← normalize(W   v)
  3. Computes ``sigma = u^T W v`` and writes ``weight = W / sigma``.

The buffers ``u`` / ``v`` are recomputed in training mode and held fixed
in eval mode — bit-identical to the reference framework's behaviour.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter

# Sentinel attribute used by ``remove_spectral_norm`` to find an active
# registration on a module.
_SN_HOOK_ATTR: str = "_spectral_norm_hooks"


def _l2_normalize(v: Tensor, eps: float = 1e-12) -> Tensor:
    """Return ``v / max(‖v‖, eps)`` — kept tiny so it inlines into the graph."""
    norm: Tensor = lucid.sqrt((v * v).sum() + eps)
    return v / norm


def _flatten_weight(w: Tensor, dim: int) -> Tensor:
    """Reshape ``w`` to 2-D with the chosen ``dim`` as rows.

    Spectral norm is defined for matrices, so we collapse all the other
    axes into a single column dimension.  The default ``dim=0`` is correct
    for ``Linear`` / ``Conv*``: rows = output channels, columns = fan-in.
    """
    if dim != 0:
        ndim: int = len(w.shape)
        perm: list[int] = [dim] + [i for i in range(ndim) if i != dim]
        w = w.permute(perm)
    rows: int = int(w.shape[0])
    return w.reshape([rows, -1])


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Module:
    """Apply spectral normalisation to ``module``'s ``name`` parameter in place.

    After the call, ``module.{name}`` is no longer a leaf parameter — it's
    a derived tensor recomputed before each forward as ``W / σ(W)``.  The
    underlying matrix lives at ``module.{name}_orig``; the power-iteration
    state lives in non-trainable buffers ``module.{name}_u`` / ``{name}_v``.

    ``dim`` selects which axis of ``W`` to treat as the row dimension; if
    omitted it defaults to 0 for everything except ``ConvTranspose*``,
    which uses the input-channel axis (1) instead.
    """
    if not isinstance(module, Module):
        raise TypeError(f"spectral_norm requires a Module, got {type(module).__name__}")
    if not hasattr(module, name):
        raise AttributeError(f"module has no parameter '{name}'")
    weight: Parameter = getattr(module, name)
    if not isinstance(weight, Parameter):
        raise TypeError(
            f"'{name}' must be a Parameter to apply spectral_norm, got "
            f"{type(weight).__name__}"
        )

    # ConvTranspose* stores weights as (in_channels, out_channels, ...) — the
    # spectral norm should be taken with the output-channel axis (1) on top.
    if dim is None:
        cls_name: str = type(module).__name__
        dim = 1 if cls_name.startswith("ConvTranspose") else 0
    if dim < 0 or dim >= len(weight.shape):
        raise ValueError(
            f"dim {dim} out of range for weight of shape {tuple(weight.shape)}"
        )

    # Move the original weight to ``{name}_orig`` and seed the iteration
    # vectors.  ``u`` is the row-side iterate (size = rows of the flattened
    # weight); ``v`` is the column-side iterate.
    weight_mat: Tensor = _flatten_weight(weight.detach(), dim)
    rows: int = int(weight_mat.shape[0])
    cols: int = int(weight_mat.shape[1])
    u_init: Tensor = _l2_normalize(lucid.randn(rows), eps=eps)
    v_init: Tensor = _l2_normalize(lucid.randn(cols), eps=eps)

    del module._parameters[name]
    module.register_parameter(name + "_orig", Parameter(weight.detach()))
    module.register_buffer(name + "_u", u_init)
    module.register_buffer(name + "_v", v_init)

    def _pre_hook(mod: Module, inputs: object) -> None:  # noqa: ANN401
        w_orig: Parameter = getattr(mod, name + "_orig")
        u: Tensor = getattr(mod, name + "_u")
        v: Tensor = getattr(mod, name + "_v")
        w_mat: Tensor = _flatten_weight(w_orig, dim)

        # ``matmul`` only handles ≥ 2-D inputs; treat each iterate as a
        # column vector ``(N, 1)`` and squeeze the trailing axis after.
        def _mv(mat: Tensor, vec: Tensor) -> Tensor:
            col: Tensor = vec.reshape([int(vec.shape[0]), 1])
            return lucid.matmul(mat, col).reshape([-1])

        # Power iteration runs in no-grad: u/v are buffers, not graph nodes.
        with lucid.no_grad():
            iters: int = n_power_iterations if mod.training else 0
            for _ in range(iters):
                v = _l2_normalize(_mv(w_mat.mT, u), eps=eps)
                u = _l2_normalize(_mv(w_mat, v), eps=eps)
            # Persist the refreshed iterates so the next forward picks up
            # where this one left off.
            if iters > 0:
                setattr(mod, name + "_u", u)
                setattr(mod, name + "_v", v)

        # ``sigma = u^T (W v)`` — keep the autograd graph here so the
        # gradient w.r.t. ``weight_orig`` flows through the normalisation.
        sigma: Tensor = (u * _mv(w_mat, v)).sum()
        normalised: Tensor = w_orig / sigma
        object.__setattr__(mod, name, normalised)

    handle = module.register_forward_pre_hook(_pre_hook)

    hooks: dict[str, object] = getattr(module, _SN_HOOK_ATTR, {})
    hooks[name] = (handle, dim)
    object.__setattr__(module, _SN_HOOK_ATTR, hooks)

    # Run the hook once so ``module.{name}`` has a sensible initial value
    # without waiting for the first forward call.
    _pre_hook(module, ())
    return module


def remove_spectral_norm(module: Module, name: str = "weight") -> Module:
    """Reverse ``spectral_norm``: collapse ``{name}_orig`` back into a
    plain leaf parameter ``{name}`` and drop the iteration buffers."""
    hooks: dict[str, object] = getattr(module, _SN_HOOK_ATTR, {})
    if name not in hooks:
        raise ValueError(f"spectral_norm not registered on '{name}'")
    handle, _dim = hooks.pop(name)
    handle.remove()

    w_orig: Parameter = getattr(module, name + "_orig")
    materialised: Tensor = w_orig.detach()
    del module._parameters[name + "_orig"]

    # Drop the cached non-leaf attribute so ``register_parameter`` succeeds.
    if name in module.__dict__:
        del module.__dict__[name]
    module.register_parameter(name, Parameter(materialised))

    # Buffers are stored on the module's ``_buffers`` dict — clear them.
    for buf_suffix in ("_u", "_v"):
        full_name: str = name + buf_suffix
        if hasattr(module, "_buffers") and full_name in module._buffers:
            del module._buffers[full_name]

    if not hooks:
        try:
            object.__delattr__(module, _SN_HOOK_ATTR)
        except AttributeError:
            pass
    return module
