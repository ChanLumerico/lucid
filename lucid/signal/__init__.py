"""lucid.signal — signal-processing utilities.

The standard reference framework's signal sub-package is small — just
``windows.*``.  Lucid keeps it minimal too: window functions live under
``lucid.signal.windows.<name>``, exactly as the reference does.

Per **H8** (sub-package shortcut ban), windows are *only* accessed via
``lucid.signal.windows.hann(...)``; no ``lucid.hann`` or top-level alias.
"""

from lucid.signal import windows

__all__ = ["windows"]
