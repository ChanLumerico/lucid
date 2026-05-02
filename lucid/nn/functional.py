"""lucid.nn.functional — fused and utility neural-network ops.

Forward-time fused kernels (Phase 19):
  linear_relu   — SGEMM + vDSP vrelu in a single cache pass (CPU)
  linear_gelu   — SGEMM + vForce GELU approximation (CPU)
  run_fusion_pass — apply structural backward-graph fusion

These are opt-in explicit APIs.  Call them instead of relu(linear(x))
to get the single-kernel speedup on CPU.
"""

from lucid._C import engine as _eng


def linear_relu(x, weight, bias):
    """Fused linear + ReLU: y = max(0, x @ W^T + b).

    CPU: BLAS SGEMM + vDSP vrelu (single memory pass).
    Falls back to a separate linear + relu on non-CPU or non-F32 inputs.
    """
    return _eng._fused_linear_relu(x, weight, bias)


def linear_gelu(x, weight, bias):
    """Fused linear + GELU: y = GELU(x @ W^T + b).

    CPU: BLAS SGEMM + vForce vvtanhf approximation.
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    return _eng._fused_linear_gelu(x, weight, bias)


def run_fusion_pass(tensor):
    """Apply structural op-fusion to the backward graph rooted at `tensor`.

    Detects LinearRelu, SDPA, and ConvBnRelu chains and marks them for
    fused backward execution.  Returns the number of patterns fused.
    """
    if not isinstance(tensor, _eng.TensorImpl):
        return 0
    return _eng._run_fusion_pass(tensor)
