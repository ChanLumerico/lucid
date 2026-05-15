// lucid/_C/nn/LSTM.h
//
// Autograd-aware single-layer LSTM.
//
// Inference path (no grad required): delegates to IBackend::lstm_forward,
// which uses BNNS (BNNSComputeInference) on CPU for maximum throughput.
//
// Training path: delegates to IBackend::lstm_forward_train, which uses
// hand-rolled BLAS and returns two extra Storage tensors:
//   gates_all  – shape (T, B, 4H), the pre-activation gate values for all
//                time steps; gate order is [i, f, g, o].
//   cells_all  – shape (T+1, B, H), the cell states c_0 through c_T.
// These are consumed by the BPTT backward in IBackend::lstm_backward.
//
// LstmBackward inherits directly from Node (not FuncOp) because the 7-edge
// topology {input, h0, c0, wih, whh, bih, bhh} is built manually rather than
// through the generic NaryKernel machinery.  Only the output hidden sequence
// (res[0]) carries the grad_fn; hn and cn are detached.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/Node.h"
#include "../backend/IBackend.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the single-layer LSTM.
//
// Stores the full gate and cell history so that the BPTT backward can
// reconstruct h_{t-1} = o_{t-1} * tanh(c_{t-1}) from saved gates/cells
// without needing to re-run the forward pass.
// Edge order: {input, h0, c0, wih, whh, bih, bhh} (indices 0–6).
class LUCID_API LstmBackward : public Node {
public:
    Storage saved_input;                 // Original input sequence (T, B, input_size).
    Storage saved_h0;                    // Initial hidden state (1, B, H).
    std::vector<Storage> saved_weights;  // {wih, whh, bih, bhh}.

    Storage gates_all;  // All gate pre-activations, shape (T, B, 4H).
    Storage cells_all;  // All cell states (including c_0), shape (T+1, B, H).

    backend::IBackend::LstmOpts opts;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    std::string_view name() const noexcept { return "LstmBackward"; }

    // Run the BPTT backward.  grad_out is the gradient w.r.t. the output
    // sequence.  Gradients for hn and cn are treated as zero.
    std::vector<Storage> apply(Storage grad_out) override;

    // Run the LSTM forward pass and return {output, hn, cn}.
    // When any input requires a gradient, lstm_forward_train is called and
    // a LstmBackward node is attached to the output tensor.
    // weights must be {wih, whh, bih, bhh} in that order.
    static std::vector<TensorImplPtr> forward(const TensorImplPtr& input,
                                              const TensorImplPtr& h0,
                                              const TensorImplPtr& c0,
                                              const std::vector<TensorImplPtr>& weights,
                                              const backend::IBackend::LstmOpts& opts);
};

// Public entry point: delegates to LstmBackward::forward.
LUCID_API std::vector<TensorImplPtr> lstm_op(const TensorImplPtr& input,
                                             const TensorImplPtr& h0,
                                             const TensorImplPtr& c0,
                                             const std::vector<TensorImplPtr>& weights,
                                             const backend::IBackend::LstmOpts& opts);

}  // namespace lucid
