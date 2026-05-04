// lucid/_C/nn/CTCLoss.h
// Connectionist Temporal Classification loss.
// log_probs: (T, N, C), targets: (N*S,) flat int32, input_lengths/target_lengths: (N,) int32.
// Returns per-sample losses (N,); apply reduction in Python.
#pragma once
#include "../api.h"
#include "../core/fwd.h"
namespace lucid {
LUCID_API TensorImplPtr ctc_loss_op(const TensorImplPtr& log_probs,
                                     const TensorImplPtr& targets,
                                     const TensorImplPtr& input_lengths,
                                     const TensorImplPtr& target_lengths,
                                     int blank,
                                     bool zero_infinity);
}  // namespace lucid
