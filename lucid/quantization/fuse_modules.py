"""``fuse_modules`` — collapse Conv/BN/ReLU (or Linear/ReLU) runs before quant.

Given groups of dotted module names, each run is replaced in place: the
first member becomes the fused module and the rest become
:class:`~lucid.nn.Identity` (so the model's original ``forward`` still
routes through them harmlessly).  BatchNorm is folded into the preceding
convolution's weight via the existing eval-time fusion; a trailing ReLU is
absorbed into an ``nn.intrinsic`` fused module so the activation observer
later sees the post-ReLU range.
"""

import copy
from typing import TYPE_CHECKING, cast

import lucid.nn as nn
from lucid.nn.utils.fusion import fuse_conv_bn_eval

if TYPE_CHECKING:
    from collections.abc import Sequence

_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _get_module(root: nn.Module, dotted: str) -> nn.Module:
    """Resolve a dotted submodule path to the module instance."""
    mod: nn.Module = root
    for part in dotted.split("."):
        mod = cast("nn.Module", getattr(mod, part))
    return mod


def _set_module(root: nn.Module, dotted: str, value: nn.Module) -> None:
    """Assign ``value`` at a dotted submodule path, preserving child order.

    Uses ``_modules`` directly rather than ``setattr``: ``Module.__setattr__``
    deletes then re-adds the key (moving it to the end), which would scramble
    ``Sequential`` execution order.
    """
    parts = dotted.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = cast("nn.Module", getattr(parent, part))
    leaf = parts[-1]
    if leaf in parent._modules:
        parent._modules[leaf] = value
    else:
        setattr(parent, leaf, value)


def _conv_relu(conv: nn.Module, relu: nn.Module) -> nn.Module:
    """Wrap a conv + ReLU into the matching ``nn.intrinsic`` fused module."""
    import lucid.nn.intrinsic as nni

    mapping: dict[type, type] = {
        nn.Conv1d: nni.ConvReLU1d,
        nn.Conv2d: nni.ConvReLU2d,
        nn.Conv3d: nni.ConvReLU3d,
    }
    return cast("nn.Module", mapping[type(conv)](conv, relu))


def _fuse_run(mods: Sequence[nn.Module]) -> list[nn.Module]:
    """Return replacements for a fusible run (fused first, ``Identity`` rest)."""
    import lucid.nn.intrinsic as nni

    ids = [nn.Identity() for _ in range(len(mods) - 1)]
    first = mods[0]
    # Conv (+ BN) (+ ReLU)
    if isinstance(first, _CONV_TYPES):
        conv: nn.Module = first
        rest = list(mods[1:])
        if rest and isinstance(
            rest[0], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            conv = cast("nn.Module", fuse_conv_bn_eval(conv, rest.pop(0)))
        if rest and isinstance(rest[0], nn.ReLU):
            rest.pop(0)
            return [_conv_relu(conv, nn.ReLU()), *ids]
        return [conv, *ids]
    # Linear + ReLU
    if isinstance(first, nn.Linear) and len(mods) == 2 and isinstance(mods[1], nn.ReLU):
        return [nni.LinearReLU(first, nn.ReLU()), *ids]
    raise ValueError(
        f"fuse_modules: unsupported fusion pattern {[type(m).__name__ for m in mods]}"
    )


def fuse_modules(
    model: nn.Module,
    modules_to_fuse: Sequence[Sequence[str]],
    inplace: bool = False,
) -> nn.Module:
    """Fuse the given groups of modules for quantization.

    Parameters
    ----------
    model : nn.Module
        Model to fuse (put in ``eval`` mode — BN folding is eval-time).
    modules_to_fuse : sequence of name groups
        Each group is a list of dotted module names to fuse, e.g.
        ``[["conv1", "bn1"], ["layer1.0.conv1", "layer1.0.bn1"]]``.  Supported
        runs: ``[Conv, BN]``, ``[Conv, BN, ReLU]``, ``[Conv, ReLU]``,
        ``[Linear, ReLU]``.
    inplace : bool, default False
        Fuse in place instead of on a deep copy.

    Returns
    -------
    nn.Module
        The fused model.
    """
    model = model if inplace else copy.deepcopy(model)
    for group in modules_to_fuse:
        mods = [_get_module(model, name) for name in group]
        replacements = _fuse_run(mods)
        for name, repl in zip(group, replacements):
            _set_module(model, name, repl)
    return model
