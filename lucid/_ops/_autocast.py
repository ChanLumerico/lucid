"""The autocast entry of a Python composite (LCD-500).

An engine op applies its row of the autocast policy table
(``lucid/_C/core/AmpPolicyTable.def``) once, at its entry, and runs its
body with autocast off.  A composite written in Python had neither, so
every engine op inside it applied its own row: under a Metal autocast
``local_response_norm`` met its float32 ``sum`` with a float16 ``div``
operand, and ``cosine_similarity`` of two float16 inputs normalised them
in float16 and only reduced in float32.

The one Python owner of that entry.  It lives here, below ``nn``,
``linalg`` and the composites, so any of them can import it; the cast rule
itself stays in the engine (``amp_cast_inputs``).
"""

from collections.abc import Iterator
from contextlib import contextmanager

from lucid._C import engine as _C_engine


@contextmanager
def _autocast_entry(
    name: str, *inputs: _C_engine.TensorImpl
) -> Iterator[list[_C_engine.TensorImpl]]:
    """``inputs`` cast by ``name``'s row, with autocast off on their device inside.

    Parameters
    ----------
    name : str
        The composite's row in the policy table.
    *inputs : TensorImpl
        Its floating operands; the first one's device selects the column.

    Yields
    ------
    list of TensorImpl
        The operands at the row's dtype — unchanged outside autocast.
    """
    device = inputs[0].device
    if not _C_engine.amp_is_active(device):
        yield list(inputs)
        return
    cast = _C_engine.amp_cast_inputs(name, list(inputs))
    guard = _C_engine.AutocastGuard(_C_engine.F32, device, False)
    try:
        yield cast
    finally:
        del guard  # the last reference — ~AutocastGuard restores the prior state
