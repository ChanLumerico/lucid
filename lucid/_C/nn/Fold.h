// lucid/_C/nn/Fold.h
// fold (col2im): inverse of unfold. (N, C*kH*kW, L) → (N, C, outH, outW).
#pragma once
#include <vector>
#include "../api.h"
#include "../core/fwd.h"
namespace lucid {
LUCID_API TensorImplPtr fold_op(const TensorImplPtr& x,
                                  const std::vector<int>& output_size,
                                  const std::vector<int>& kernel_size,
                                  const std::vector<int>& stride,
                                  const std::vector<int>& padding,
                                  const std::vector<int>& dilation);
}  // namespace lucid
