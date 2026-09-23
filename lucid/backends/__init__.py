"""
lucid.backends: backend configuration flags.

Mirrors ``backend configuration`` semantics for Apple Silicon:

    lucid.backends.accelerate.deterministic = True   # CPU BLAS reproducibility
    lucid.backends.metal.deterministic = True        # GPU Metal reproducibility
    lucid.backends.metal.benchmark = False           # reserved for future use
"""

from typing import final

from lucid._C import engine as _C_engine


@final
class _AccelerateBackend:
    """Apple Accelerate (CPU) backend settings."""

    @property
    def deterministic(self) -> bool:
        """If True, force deterministic algorithms in CPU kernels."""
        return _C_engine.is_deterministic()

    @deterministic.setter
    def deterministic(self, value: bool) -> None:
        _C_engine.set_deterministic(value)


@final
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


@final
class _QuantizedBackend:
    """Low-precision (int4/int8) GEMM backend selection.

    ``engine`` chooses how quantized modules run their matmul:

    * ``"auto"`` (default) — use the MLX group-wise quantized kernel when the
      engine exposes it (``available`` is True) and the tensors are on Metal;
      otherwise fall back to the dequantize-to-float path.
    * ``"mlx_group"`` — force the MLX quantized kernel (errors if unavailable).
    * ``"reference"`` — always dequantize-to-float (portable, slower).
    """

    _engine: str = "auto"

    @property
    def available(self) -> bool:
        """True if the engine was built with the MLX quantized-GEMM ops."""
        return hasattr(_C_engine, "quantized")

    @property
    def engine(self) -> str:
        """Selected quantized-GEMM path (``"auto"`` / ``"mlx_group"`` / ``"reference"``)."""
        return self._engine

    @engine.setter
    def engine(self, value: str) -> None:
        if value not in ("auto", "mlx_group", "reference"):
            raise ValueError(
                f"quantized.engine must be 'auto'/'mlx_group'/'reference', got {value!r}"
            )
        self._engine = value

    def use_mlx(self) -> bool:
        """Whether the MLX quantized kernel should run given the current mode."""
        if self._engine == "reference":
            return False
        if self._engine == "mlx_group":
            return True
        return self.available  # "auto"


accelerate = _AccelerateBackend()
metal = _MetalBackend()
quantized = _QuantizedBackend()

__all__ = ["accelerate", "metal", "quantized"]
