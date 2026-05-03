// lucid/_C/nn/AdaptivePool.h
//
// Adaptive max-pooling and average-pooling for 1-D, 2-D, and 3-D inputs.
// Each function computes the required kernel size from the ratio of the input
// spatial dimension to the requested output size, then delegates to the
// corresponding fixed-stride pooling op in PoolNd.h.
//
// Current restriction: every spatial input dimension must be evenly divisible
// by the corresponding output dimension.  Non-uniform partitions are not yet
// implemented; a descriptive not_implemented exception is thrown if violated.

#pragma once

#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Adaptive max-pool for 3-D input (B, C, L) to output length OL.
// Requires L % OL == 0; kernel = L / OL, stride = kernel, padding = 0.
LUCID_API TensorImplPtr adaptive_max_pool1d_op(const TensorImplPtr& x, int OL);

// Adaptive max-pool for 4-D input (B, C, H, W) to output (OH, OW).
LUCID_API TensorImplPtr adaptive_max_pool2d_op(const TensorImplPtr& x, int OH, int OW);

// Adaptive max-pool for 5-D input (B, C, D, H, W) to output (OD, OH, OW).
LUCID_API TensorImplPtr adaptive_max_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

// Adaptive average-pool for 3-D input (B, C, L) to output length OL.
LUCID_API TensorImplPtr adaptive_avg_pool1d_op(const TensorImplPtr& x, int OL);

// Adaptive average-pool for 4-D input (B, C, H, W) to output (OH, OW).
LUCID_API TensorImplPtr adaptive_avg_pool2d_op(const TensorImplPtr& x, int OH, int OW);

// Adaptive average-pool for 5-D input (B, C, D, H, W) to output (OD, OH, OW).
LUCID_API TensorImplPtr adaptive_avg_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

}  // namespace lucid
