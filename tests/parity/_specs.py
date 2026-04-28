"""
OpSpec — single declarative description of one engine op for parity testing.

A spec is the *only* thing you write to add an op to the harness; everything
else (CPU/GPU forward parity, autograd parity, cross-device parity) is
derived. See `_harness.py` for what each axis actually does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


# Type aliases for clarity.
NpArrays = Sequence["np.ndarray"]  # noqa: F821  — np.ndarray imported lazily
TensorImpls = Sequence[Any]         # actual type: List[engine.TensorImpl]


@dataclass
class OpSpec:
    """One op = one spec. The harness derives 6 tests from it.

    Required:
        name:           Pytest id ("matmul_2x3_3x4"). Must be unique.
        engine_fn:      (List[TensorImpl], **kwargs) -> TensorImpl.
                        Receives the engine inputs already on the right device.
        torch_fn:       (List[torch.Tensor], **kwargs) -> torch.Tensor.
                        Same arg structure for direct comparison.
        input_shapes:   List of input shapes. The harness allocates one tensor
                        per shape, all of dtype=`dtype` and seeded by `seed`.

    Optional:
        dtype:          Numpy dtype. Default float32.
        kwargs:         Static keyword args forwarded to BOTH fn's verbatim.
        atol / rtol:    Float tolerances. Defaults sized for F32 GPU paths.
        seed:           RNG seed for reproducibility.
        skip_gpu:       True for ops that genuinely can't live on GPU
                        (e.g. F64 histogram). Cross-device test is also skipped.
        skip_grad:      True for non-differentiable ops (argmax, comparisons,
                        bitwise, where indexing dominates).
        input_gen:      If reproducibility-via-seed isn't enough — e.g. SPD
                        matrices for cholesky — pass a fn (rng) -> List[np.ndarray].
                        When set, `input_shapes` and `dtype` are ignored.
        post_fn:        Reduce-to-scalar function for backward. The harness
                        sums by default; specify if a different reduction is
                        needed for a meaningful gradient signal.
        notes:          Free-form. Surfaced in failure messages.
    """
    name: str
    engine_fn: Callable[..., Any]
    torch_fn:  Callable[..., Any]
    input_shapes: Sequence[Sequence[int]] = ()
    dtype: str = "float32"
    kwargs: dict = field(default_factory=dict)
    atol: float = 1e-4
    rtol: float = 1e-4
    seed: int = 1234
    skip_gpu: bool = False
    skip_grad: bool = False
    input_gen: Callable[..., Any] | None = None
    post_fn: Callable[..., Any] | None = None
    notes: str = ""


def collect_specs(modules: Iterable) -> list[OpSpec]:
    """Aggregate `SPECS: list[OpSpec]` from a list of imported modules."""
    out: list[OpSpec] = []
    for m in modules:
        out.extend(getattr(m, "SPECS", []))
    return out
