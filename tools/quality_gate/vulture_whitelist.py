"""Names vulture cannot see used, listed so they are not counted as dead.

vulture reads this file as Python and treats every name in it as used.
Add a name here only when something outside Python's view calls it — the
C++ engine through pybind11, a lookup by string, a protocol the runtime
invokes — and say which on the same line.  A name that is merely unused
belongs in a deletion, not here.

Dunders are already ignored by the gate (``--ignore-names "__*__"``).
"""
