"""Math constants surfaced at the top level of ``lucid``.

These are plain Python floats — keeping them in the composite layer means
the top-level namespace doesn't grow a special case for "names that aren't
callables".
"""

import math

pi: float = math.pi
e: float = math.e
inf: float = math.inf
nan: float = math.nan

# ``newaxis`` is ``None`` to match NumPy / PyTorch indexing idioms:
# ``x[:, lucid.newaxis]`` works because ``None`` already means "insert
# a singleton axis here" in our advanced-indexing path.
newaxis = None  # type: ignore[assignment]


__all__ = ["pi", "e", "inf", "nan", "newaxis"]
