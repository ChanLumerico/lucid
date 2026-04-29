"""Lucid C++ engine — Python entry point.

Performs an ABI version check on import so users get a clear error if they
mix a stale .so with a newer header (or vice versa).
"""

_EXPECTED_ABI = 8  # must match LUCID_ABI_VERSION in lucid/_C/version.h

from lucid._C import engine as _engine  # noqa: E402

_actual = getattr(_engine, "ABI_VERSION", None)
if _actual != _EXPECTED_ABI:
    raise ImportError(
        f"Lucid ABI version mismatch: Python wrapper expects ABI {_EXPECTED_ABI} "
        f"but loaded .so reports ABI {_actual}. Rebuild the C++ extension."
    )

del _actual
