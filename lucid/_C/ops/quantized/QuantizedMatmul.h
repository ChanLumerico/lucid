// lucid/_C/ops/quantized/QuantizedMatmul.h
//
// GPU-stream low-precision (int4 / int8) matmul primitives, thin wrappers over
// MLX's group-wise affine quantization kernels.  Inference-only (no autograd) —
// they exist so a converted quantized model can execute a *real* low-precision
// GEMM instead of the dequantize-to-float path, which is where quantization's
// compute + memory win lives on Apple Silicon.
//
// GPU stream only (H3): all three ops require Device::GPU inputs and run on the
// default MLX stream.  The packed weight returned by ``quantize_op`` is MLX's
// uint32-packed layout; Lucid has no uint32 dtype, so it is tagged ``I32``
// (same 32-bit width) and treated as an opaque buffer — only ever handed back
// to ``dequantize_op`` / ``quantized_matmul_op``.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Quantize a float weight to MLX group-wise affine.  Returns
// ``{packed_weight (I32-tagged uint32), scales (F32), biases (F32)}``.
LUCID_API std::vector<TensorImplPtr> quantize_op(
    const TensorImplPtr& w, int group_size, int bits);

// Reconstruct a float weight from its packed form (inverse of quantize_op).
LUCID_API TensorImplPtr dequantize_op(
    const TensorImplPtr& w, const TensorImplPtr& scales, const TensorImplPtr& biases,
    int group_size, int bits);

// ``x @ w`` where ``w`` is packed (from quantize_op).  ``transpose`` matches
// MLX semantics (weight stored as ``(out, in)`` → ``transpose=true``).
LUCID_API TensorImplPtr quantized_matmul_op(
    const TensorImplPtr& x, const TensorImplPtr& w, const TensorImplPtr& scales,
    const TensorImplPtr& biases, bool transpose, int group_size, int bits);

}  // namespace lucid
