"""
lucid.backends: backend configuration flags.

Mirrors ``backend configuration`` semantics for Apple Silicon:

    lucid.backends.accelerate.deterministic = True   # CPU BLAS reproducibility
    lucid.backends.metal.deterministic = True        # GPU Metal reproducibility
    lucid.backends.metal.benchmark = False           # reserved for future use
"""

from lucid._C import engine as _C_engine


class _AccelerateBackend:
    """Apple Accelerate (CPU) backend settings."""

    @property
    def deterministic(self) -> bool:
        """If True, force deterministic algorithms in CPU kernels."""
        return _C_engine.is_deterministic()

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        _C_engine.set_deterministic(value)


class _MetalBackend:
    """Apple Metal (GPU) backend settings."""

    # benchmark: controls whether MLX auto-tunes kernel parameters.
    # MLX does not currently expose a tuning API; kept for API compatibility.
    _benchmark: bool = False

    @property
    def deterministic(self) -> bool:
        """If True, force deterministic algorithms in GPU (MLX) kernels."""
        return _C_engine.is_deterministic()

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        _C_engine.set_deterministic(value)

    @property
    def benchmark(self) -> bool:
        """Kernel auto-tuning hint (reserved; MLX manages this internally)."""
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value: bool) -> None:
        self._benchmark = value


accelerate = _AccelerateBackend()
metal = _MetalBackend()

__all__ = ["accelerate", "metal"]
