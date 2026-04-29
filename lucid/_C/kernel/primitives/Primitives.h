#pragma once

// =====================================================================
// Lucid C++ engine — kernel/primitives/Primitives.h
// =====================================================================
//
// Convenience umbrella — includes all kernel compute primitives.
//
// Individual headers:
//   BatchedMatmul.h   — N-D batched matrix multiply (CPU: sgemm/dgemm)
//   BroadcastReduce.h — gradient sum-reduce to undo NumPy broadcasting
//   Gather.h          — N-D coordinate gather / flat-index lookup
//   Im2Col.h          — im2col / col2im for convolution ops
//   Scatter.h         — multi-corner weighted scatter-add (interp bwd)
//
// Layer: kernel/primitives/ (rank 4).
// May include: core/ (0), tensor/ (1), backend/ (2), autograd/ (3).

#include "BatchedMatmul.h"
#include "BroadcastReduce.h"
#include "Gather.h"
#include "Im2Col.h"
#include "Scatter.h"
