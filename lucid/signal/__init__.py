"""lucid.signal — signal-processing utilities.

The standard reference framework's signal sub-package is small — just
``windows.*``.  Lucid keeps it minimal too: window functions live under
``lucid.signal.windows.<name>``, exactly as the reference does, and that
module is the one place the window math is written.

Per **H8** (sub-package shortcut ban), these windows are *only* accessed
via ``lucid.signal.windows.hann(...)``; there is no ``lucid.hann`` alias.
The top-level ``lucid.hann_window`` family and ``lucid.stft`` /
``lucid.istft`` are not aliases of this package but the reference's own
top-level functions (a ``periodic`` flag, ``window_length``); the
factories forward to the windows here.
"""

from lucid.signal import windows

__all__ = ["windows"]
