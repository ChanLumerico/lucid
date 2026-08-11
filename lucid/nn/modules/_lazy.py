"""Filling a lazy layer's placeholder parameters in place.

A lazy layer cannot build its weights until it sees an input.  What it
*can* do is exist: registering an
:class:`~lucid.nn.parameter.UninitializedParameter` up front means that
anything reading ``parameters()`` before the first forward — an
optimizer, an EMA, a parameter-server shard — is handed the objects the
real weights will later occupy, instead of an empty list it keeps
forever.

``fill`` is what turns one into the other.  It exists as a shared helper
because thirteen lazy classes across three files have to do exactly the
same thing, and the failure mode of each of them doing it separately is
that one of them assigns instead of materializing and quietly reverts to
the old behaviour for that layer alone.
"""

from typing import TYPE_CHECKING

from lucid.nn.parameter import Parameter, UninitializedParameter

if TYPE_CHECKING:
    from lucid.nn.module import Module


def fill(module: Module, name: str, value: Parameter) -> None:
    """Give ``module.<name>`` its real value without replacing the object.

    Parameters
    ----------
    module : Module
        The lazy layer being initialised.
    name : str
        Which registered parameter to fill.
    value : Parameter
        The real parameter — correct shape, already initialised.

    Notes
    -----
    Falls back to plain assignment when the slot does not hold a
    placeholder.  That happens on the second materialisation of a layer
    loaded from a checkpoint and then run, and there is nothing to
    preserve in that case; assigning is right.
    """
    current = module._parameters.get(name)
    if isinstance(current, UninitializedParameter):
        current.materialize(value)
    else:
        setattr(module, name, value)


__all__ = ["fill"]
