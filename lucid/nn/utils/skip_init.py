"""``lucid.nn.utils.skip_init`` — instantiate a module without weight
initialisation.

Useful when loading a checkpoint immediately after construction: avoids
paying the cost of ``xavier_uniform`` / ``kaiming_normal`` / etc. only to
overwrite the tensors a moment later.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.nn.module import Module


def skip_init(module_cls: type, *args: object, **kwargs: object) -> Module:
    r"""Construct a module while skipping its parameter initialisation.

    Instantiates ``module_cls(*args, **kwargs)`` with every
    :mod:`lucid.nn.init` initialiser turned into a no-op, so each
    parameter keeps whatever its allocation left.  The intended pattern
    is "construct, then ``load_state_dict``": skipping the
    :func:`~lucid.nn.init.xavier_uniform_` /
    :func:`~lucid.nn.init.kaiming_normal_` work that is about to be
    overwritten saves measurable time on large models with many layers.

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
    The initialisers are skipped, not run and discarded: no random draw
    and no fill happens while the module is built, and the global random
    state is left where it was.  A module that fills its parameters some
    other way — by assigning a freshly drawn tensor in ``__init__`` —
    still pays for that.

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
    # The initialisers are no-ops while the module is built, so its
    # parameters keep whatever their allocation left and no random draw or
    # fill runs.  It used to construct normally and then swap each
    # parameter for fresh empty storage: every initialiser still ran, and
    # the model was briefly allocated twice.
    from lucid.nn.init import _skipping

    with _skipping():
        module: Module = module_cls(*args, **kwargs)
    return module
