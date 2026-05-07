"""Pruning utilities — minimal entry-point parity with the reference framework.

The reference framework's ``nn.utils.prune`` is a large module with many
pruning methods (``L1Unstructured``, ``RandomStructured``, ``RandomUnstructured``,
``LnStructured``, custom pruners, ``global_unstructured`` ...).  We expose
the most-used pieces — ``identity`` / ``random_unstructured`` /
``l1_unstructured`` / ``remove`` / ``is_pruned`` — as a thin layer on top of
forward-pre-hooks so common ``prune.l1_unstructured(layer, 'weight', amount=0.5)``
usage works.  More exotic methods raise ``NotImplementedError`` rather than
silently misbehaving.

The mask is registered as a buffer ``{name}_mask``; the original Parameter
is preserved as ``{name}_orig`` (matching upstream's naming so checkpoints
are wire-compatible).  Each forward call rebuilds ``{name} = {name}_orig *
{name}_mask`` via a pre-hook.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter

# Sentinel attribute mapping ``tensor_name`` → handle so ``remove`` /
# ``is_pruned`` can find the registration.
_PRUNE_HOOK_ATTR: str = "_prune_hooks"


def _install_mask(module: Module, name: str, mask: Tensor) -> Module:
    """Shared backend for the various pruning methods.

    Captures the original Parameter as ``{name}_orig``, registers ``mask``
    as a buffer ``{name}_mask``, and installs a forward-pre-hook that
    materialises ``{name} = {name}_orig * {name}_mask`` before each call.
    """
    if not hasattr(module, name):
        raise AttributeError(f"module has no parameter '{name}'")
    weight: Parameter = getattr(module, name)
    if not isinstance(weight, Parameter):
        raise TypeError(
            f"'{name}' must be a Parameter to prune, got {type(weight).__name__}"
        )
    if tuple(mask.shape) != tuple(weight.shape):
        raise ValueError(
            f"prune mask shape {tuple(mask.shape)} does not match weight "
            f"shape {tuple(weight.shape)}"
        )

    # Preserve the original; install the mask as a non-trainable buffer.
    del module._parameters[name]
    module.register_parameter(name + "_orig", Parameter(weight.detach()))
    module.register_buffer(name + "_mask", mask)

    def _pre_hook(mod: Module, inputs: object) -> None:  # noqa: ANN401
        w: Parameter = getattr(mod, name + "_orig")
        m: Tensor = getattr(mod, name + "_mask")
        object.__setattr__(mod, name, w * m)

    handle = module.register_forward_pre_hook(_pre_hook)
    hooks: dict[str, object] = getattr(module, _PRUNE_HOOK_ATTR, {})
    hooks[name] = handle
    object.__setattr__(module, _PRUNE_HOOK_ATTR, hooks)

    # Materialise ``{name}`` immediately so attribute access works without
    # waiting for the first forward.
    object.__setattr__(module, name, weight.detach() * mask)
    return module


def identity(module: Module, name: str = "weight") -> Module:
    """Apply a no-op pruning (mask of all ones).

    Useful as a starting point that other pruning methods can layer on top
    of, and as the simplest sanity check that the prune machinery is wired
    up correctly.
    """
    weight: Parameter = getattr(module, name)
    mask: Tensor = lucid.ones_like(weight)
    return _install_mask(module, name, mask)


def random_unstructured(
    module: Module, name: str = "weight", amount: float = 0.5
) -> Module:
    """Mask out a random ``amount`` fraction of elements (Bernoulli).

    Each element is independently selected for pruning with probability
    ``amount``.  Surviving elements stay at their original value (no
    rescaling — matches the reference framework's convention; rescaling
    is the user's responsibility if needed).
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}")
    weight: Parameter = getattr(module, name)
    rand_t: Tensor = lucid.rand(*weight.shape, dtype=weight.dtype)
    mask: Tensor = rand_t >= amount  # True where we keep
    # bool -> dtype conversion via where + ones/zeros so multiplication works.
    mask = lucid.where(mask, lucid.ones_like(weight), lucid.zeros_like(weight))
    return _install_mask(module, name, mask)


def l1_unstructured(
    module: Module, name: str = "weight", amount: float = 0.5
) -> Module:
    """Mask out the smallest-magnitude ``amount`` fraction of elements.

    Sorts the absolute values, picks the threshold at the ``amount``-th
    quantile, and keeps everything ≥ that threshold.  Common starting
    point for magnitude-based pruning of trained weights.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}")
    weight: Parameter = getattr(module, name)
    abs_w: Tensor = lucid.abs(weight.detach())
    flat: Tensor = abs_w.reshape([-1])
    n_total: int = int(flat._impl.numel())
    n_drop: int = int(round(amount * n_total))
    if n_drop == 0:
        # Identity behaviour when amount rounds to nothing.
        return identity(module, name)

    # The reference framework uses the (n_drop)-th smallest as the cutoff;
    # ``kthvalue`` returns the k-th smallest (1-indexed), so n_drop is the
    # right argument.  The bare comparison op doesn't broadcast a 0-d
    # tensor against a multi-D one, so materialise the threshold as a
    # ``full_like`` tensor that already shares the weight's shape.
    threshold_scalar: float = float(lucid.kthvalue(flat, n_drop).item())
    threshold_t: Tensor = lucid.full_like(abs_w, threshold_scalar)
    keep: Tensor = abs_w > threshold_t
    mask: Tensor = lucid.where(keep, lucid.ones_like(weight), lucid.zeros_like(weight))
    return _install_mask(module, name, mask)


def remove(module: Module, name: str = "weight") -> Module:
    """Strip the pruning machinery from ``module.<name>``.

    The current pruned values become a fresh leaf Parameter; the
    ``{name}_orig`` and ``{name}_mask`` slots are dropped.  Forward pre-hook
    is detached.
    """
    hooks: dict[str, object] = getattr(module, _PRUNE_HOOK_ATTR, {})
    if name not in hooks:
        raise ValueError(f"prune not registered on '{name}'")
    handle = hooks.pop(name)
    handle.remove()

    w_orig: Parameter = getattr(module, name + "_orig")
    mask: Tensor = getattr(module, name + "_mask")
    final: Tensor = (w_orig.detach() * mask).detach()
    del module._parameters[name + "_orig"]
    if name + "_mask" in module._buffers:
        del module._buffers[name + "_mask"]
    if name in module.__dict__:
        del module.__dict__[name]
    module.register_parameter(name, Parameter(final))

    if not hooks:
        try:
            object.__delattr__(module, _PRUNE_HOOK_ATTR)
        except AttributeError:
            pass
    return module


def is_pruned(module: Module) -> bool:
    """Return True if ``module`` has any active pruning registration."""
    hooks: dict[str, object] = getattr(module, _PRUNE_HOOK_ATTR, {})
    return bool(hooks)
