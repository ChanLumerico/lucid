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
    """Create an instance of ``module_cls`` with **uninitialized** parameters.

    All learnable parameters are replaced with ``lucid.empty`` tensors of the
    same shape, dtype, and device immediately after construction.  Buffer
    tensors are left intact (they are typically non-learnable state such as
    running statistics that need a defined initial value).

    Parameters
    ----------
    module_cls : type
        A :class:`~lucid.nn.Module` subclass to instantiate.
    *args, **kwargs
        Forwarded verbatim to ``module_cls.__init__``.

    Returns
    -------
    Module
        The constructed module with uninitialized parameter data.

    Notes
    -----
    Unlike the reference framework's implementation (which avoids allocation
    via a ``meta`` device), Lucid allocates parameters normally and then
    replaces the underlying storage with an uninitialised buffer.  The
    semantic effect is identical: callers must not read parameter values
    before loading a checkpoint.
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
