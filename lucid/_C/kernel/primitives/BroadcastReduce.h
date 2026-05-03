// lucid/_C/kernel/primitives/BroadcastReduce.h
//
// Re-export of the autograd broadcast-reduction helpers into the kernel
// primitives namespace. The primary helper exposed here is
// reduce_grad_to_shape(), which sums gradient components over the
// dimensions that were broadcast during the forward pass, restoring the
// gradient to the original pre-broadcast input shape. It is used in the
// apply() implementations of BinaryKernel and UnaryKernel, and by ops
// that manually manage gradient accumulation. The implementation lives in
// autograd/Helpers.h.

#pragma once

#include "../../autograd/Helpers.h"
