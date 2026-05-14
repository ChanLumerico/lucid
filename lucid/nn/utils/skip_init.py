"""``lucid.nn.utils.skip_init`` — instantiate a module without weight
initialisation.

Useful when loading a checkpoint immediately after construction: avoids
paying the cost of ``xavier_uniform`` / ``kaiming_normal`` / etc. only to
overwrite the tensors a moment later.
"""

from typing import TYPE_CHECKING

import lucid
import lucid.nn as nn

if TYPE_CHECKING:
    from lucid.nn.module import Module


def skip_init(module_cls: type, *args: object, **kwargs: object) -> Module:
    r"""Construct a module while skipping its parameter initialisation.

    Instantiates ``module_cls(*args, **kwargs)`` and then immediately
    replaces every learnable parameter with an *uninitialised* buffer
    of the same shape / dtype / device.  The intended pattern is
    "construct, then ``load_state_dict``": skipping the
    :func:`xavier_uniform` / :func:`kaiming_normal` work that is about
    to be overwritten saves measurable time on large models with many
    layers.

    Parameters
    ----------
    module_cls : type
        A :class:`~lucid.nn.Module` subclass to instantiate.
    *args
        Positional arguments forwarded to ``module_cls.__init__``.
    **kwargs
        Keyword arguments forwarded to ``module_cls.__init__``.

    Returns
    -------
    Module
        Fully constructed module whose parameter data is undefined.
        Buffers (running statistics, attention masks, etc.) are left
        intact because they are typically not refreshed from a
        checkpoint.

    Notes
    -----
    Lucid allocates each parameter normally, then swaps its underlying
    storage for the output of :func:`lucid.empty` — semantically
    equivalent to the reference framework's ``meta``-device trick.
    Callers **must not** inspect parameter values before loading a
    checkpoint; the contents are unspecified and may contain ``nan``
    or ``inf``.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import skip_init
    >>> model = skip_init(nn.Linear, 1024, 1024)
    >>> # ... immediately load a checkpoint ...
    """
    module: Module = module_cls(*args, **kwargs)

    # Walk every named parameter and swap its storage for an empty tensor.
    for name, param in list(module.named_parameters()):
        # Navigate to the immediate parent module of this leaf parameter.
        parts: list[str] = name.split(".")
        parent: Module = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name: str = parts[-1]

        # Build a new Parameter with uninitialised (empty) data.
        new_param = nn.Parameter(
            lucid.empty(tuple(param.shape), dtype=param.dtype, device=param.device),
            requires_grad=param.requires_grad,
        )
        setattr(parent, leaf_name, new_param)

    return module
