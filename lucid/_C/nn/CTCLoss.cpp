// lucid/_C/nn/CTCLoss.cpp
#include "CTCLoss.h"
#include "../backend/Dispatcher.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
namespace lucid {

TensorImplPtr ctc_loss_op(const TensorImplPtr& log_probs,
                           const TensorImplPtr& targets,
                           const TensorImplPtr& input_lengths,
                           const TensorImplPtr& target_lengths,
                           int blank,
                           bool zero_infinity) {
    Validator::input(log_probs,      "ctc_loss.log_probs").float_only().non_null();
    Validator::input(targets,        "ctc_loss.targets").non_null();
    Validator::input(input_lengths,  "ctc_loss.input_lengths").non_null();
    Validator::input(target_lengths, "ctc_loss.target_lengths").non_null();

    const auto& lp_shape = log_probs->shape();
    if (lp_shape.size() != 3)
        ErrorBuilder("ctc_loss").fail("log_probs must be 3-D (T, N, C)");

    const int N = static_cast<int>(lp_shape[1]);
    Shape out_shape{static_cast<std::int64_t>(N)};

    auto& be = backend::Dispatcher::for_device(log_probs->device());
    Storage out = be.ctc_loss_forward(
        log_probs->storage(), targets->storage(),
        input_lengths->storage(), target_lengths->storage(),
        lp_shape, blank, zero_infinity, log_probs->dtype());

    return std::make_shared<TensorImpl>(
        std::move(out), out_shape, log_probs->dtype(), log_probs->device(), false);
}

}  // namespace lucid
