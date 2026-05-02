#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/Node.h"
#include "../backend/IBackend.h"
#include "../core/Dtype.h"
#include "../core/Device.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// ---------------------------------------------------------------------------
// LstmBackward — autograd node for the single-layer LSTM forward.
//
// Saved during forward:
//   saved_input    (T, B, I)
//   saved_h0       (B, H)
//   saved_weights  [wih(4H,I), whh(4H,H), bih(4H), bhh(4H)]
//   gates_all      (T, B, 4H) — i,f,g,o gate activations per step
//   cells_all      (T+1, B, H) — c0, c1, …, cT
//
// apply(grad_out) runs BPTT and returns 6 gradients:
//   {dX, dh0, dc0, dWih, dWhh, dB}  (dBih == dBhh == dB/2 combined)
// ---------------------------------------------------------------------------
class LUCID_API LstmBackward : public Node {
public:
    Storage saved_input;
    Storage saved_h0;
    std::vector<Storage> saved_weights;  // wih, whh, bih, bhh

    Storage gates_all;
    Storage cells_all;

    backend::IBackend::LstmOpts opts;
    Dtype  dtype  = Dtype::F32;
    Device device = Device::CPU;

    std::string_view name() const noexcept { return "LstmBackward"; }

    std::vector<Storage> apply(Storage grad_out) override;

    // Entry point for the autograd-aware LSTM forward.
    // Returns {output, h_n, c_n}  (only output has grad_fn set).
    static std::vector<TensorImplPtr> forward(
        const TensorImplPtr& input,
        const TensorImplPtr& h0,
        const TensorImplPtr& c0,
        const std::vector<TensorImplPtr>& weights,
        const backend::IBackend::LstmOpts& opts);
};

// Free function: autograd-aware LSTM. Returns {output, h_n, c_n}.
LUCID_API std::vector<TensorImplPtr> lstm_op(
    const TensorImplPtr& input,
    const TensorImplPtr& h0,
    const TensorImplPtr& c0,
    const std::vector<TensorImplPtr>& weights,
    const backend::IBackend::LstmOpts& opts);

}  // namespace lucid
