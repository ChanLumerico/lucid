#pragma once

// =====================================================================
// Lucid C++ engine — N-D adaptive pooling (uniform-stride case).
// =====================================================================
//
//   adaptive_{max,avg}_pool{1,2,3}d(x, output_size)
//     x : (B, C, *S)
//     y : (B, C, *O) where O is the user-specified output_size
//
// In the uniform case (S[i] % O[i] == 0 for every spatial axis),
// adaptive pooling is exactly regular pooling with
//     kernel[i] = stride[i] = S[i] / O[i],  padding = 0.
// We delegate to the corresponding {Max,Avg}Pool{N}dBackward — no new
// op-level state, no new CPU/GPU kernels. The non-uniform case
// (where S/O is fractional and per-output-cell windows have varying
// extent) is rejected with a clear NotImplementedError; users can
// either pad x to make it divisible, or fall back to regular pooling.
//
// AMP policy: KeepInput (inherited from underlying pool ops).

#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr adaptive_max_pool1d_op(const TensorImplPtr& x, int OL);
LUCID_API TensorImplPtr adaptive_max_pool2d_op(const TensorImplPtr& x, int OH, int OW);
LUCID_API TensorImplPtr adaptive_max_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);
LUCID_API TensorImplPtr adaptive_avg_pool1d_op(const TensorImplPtr& x, int OL);
LUCID_API TensorImplPtr adaptive_avg_pool2d_op(const TensorImplPtr& x, int OH, int OW);
LUCID_API TensorImplPtr adaptive_avg_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

}  // namespace lucid
