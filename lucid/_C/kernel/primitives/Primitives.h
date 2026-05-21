// lucid/_C/kernel/primitives/Primitives.h
//
// Aggregated convenience include for all kernel primitive helpers.
//
// Including this single header pulls in :file:`BatchedMatmul.h`,
// :file:`BroadcastReduce.h`, :file:`Gather.h`, :file:`Im2Col.h`, and
// :file:`Scatter.h` in one shot, so that :file:`ops/` nodes can
// ``#include "kernel/primitives/Primitives.h"`` instead of enumerating
// every individual primitive header at every call site.
//
// Notes
// -----
// This header has no content of its own — it is a pure umbrella header
// and intentionally exposes no new declarations.  Adding a new primitive
// requires only (1) creating its dedicated header and (2) appending an
// ``#include`` line below; downstream callers pick it up automatically.
//
// See Also
// --------
// BatchedMatmul.h    : N-D batched GEMM helpers used by MatMul.
// BroadcastReduce.h  : ``reduce_grad_to_shape`` for broadcast-aware
//                      gradient reductions.
// Gather.h           : Index-select primitives for Embedding / upsample.
// Im2Col.h           : Convolution im2col / col2im transforms.
// Scatter.h          : Scatter-add and zero-storage helpers (adjoint of
//                      Gather).

#pragma once

#include "BatchedMatmul.h"
#include "BroadcastReduce.h"
#include "Gather.h"
#include "Im2Col.h"
#include "Scatter.h"
