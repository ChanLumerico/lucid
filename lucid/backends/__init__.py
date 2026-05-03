"""
lucid.backends: backend configuration flags.
"""


class _AccelerateBackend:
    """Apple Accelerate CPU backend settings."""

    deterministic: bool = False


class _MetalBackend:
    """Apple Metal GPU backend settings."""

    benchmark: bool = False
    deterministic: bool = False


accelerate = _AccelerateBackend()
metal = _MetalBackend()

__all__ = ["accelerate", "metal"]
