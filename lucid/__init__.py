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

# Re-export the engine surface at the lucid.* namespace level.
from lucid._C.engine import (  # noqa: F401, E402
    TensorImpl,
    Device,
    Dtype,
    NoGradGuard,
    grad_enabled,
    set_grad_enabled,
    engine_backward,
    FunctionCtx,
    _PythonBackwardNode,
    _register_python_backward_node,
)
