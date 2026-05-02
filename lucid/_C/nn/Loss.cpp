#include "Loss.h"

#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

Shape reduced_shape(const Shape& in, Reduction red) {
    if (red == Reduction::None)
        return in;
    return Shape{};
}

}  // namespace

const OpSchema MseLossBackward::schema_v1{"mse_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr MseLossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       Reduction reduction) {
    if (!input || !target)
        ErrorBuilder("mse_loss").fail("null input");
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "mse_loss: input/target shape mismatch");
    if (input->dtype() != target->dtype())
        throw DtypeMismatch(std::string(dtype_name(input->dtype())),
                            std::string(dtype_name(target->dtype())), "mse_loss");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage = backend::Dispatcher::for_device(input->device())
                              .mse_loss(input->storage(), target->storage(), input->shape(),
                                        input->dtype(), static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<MseLossBackward>();
        bwd->reduction_ = reduction;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<MseLossBackward, 2>::wire_autograd(std::move(bwd), {input, target}, out);
    }
    return out;
}

std::vector<Storage> MseLossBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_).mse_loss_backward(
        saved_inputs_[0], saved_inputs_[1], grad_out, orig_shape_, dtype_,
        static_cast<int>(reduction_));
    return {std::move(grads.first), std::move(grads.second)};
}

TensorImplPtr mse_loss_op(const TensorImplPtr& input, const TensorImplPtr& target, int reduction) {
    return MseLossBackward::forward(input, target, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(MseLossBackward)

const OpSchema BCELossBackward::schema_v1{"bce_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BCELossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       const TensorImplPtr& weight,
                                       Reduction reduction,
                                       double eps) {
    if (!input || !target || !weight)
        ErrorBuilder("bce_loss")
            .fail("input/target/weight required (pass ones for weight if not used)");
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "bce_loss: input/target shape mismatch");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .bce_loss(input->storage(), target->storage(), weight->storage(), input->shape(),
                      input->dtype(), eps, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<BCELossBackward>();
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<BCELossBackward, 3>::wire_autograd(std::move(bwd),
                                                              {input, target, weight}, out);
    }
    return out;
}

std::vector<Storage> BCELossBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_).bce_loss_backward(
        saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], grad_out, orig_shape_, dtype_, eps_,
        static_cast<int>(reduction_));
}

TensorImplPtr bce_loss_op(const TensorImplPtr& input,
                          const TensorImplPtr& target,
                          const TensorImplPtr& weight,
                          int reduction,
                          double eps) {
    return BCELossBackward::forward(input, target, weight, static_cast<Reduction>(reduction), eps);
}
LUCID_REGISTER_OP(BCELossBackward)

const OpSchema BCEWithLogitsBackward::schema_v1{"bce_with_logits", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr BCEWithLogitsBackward::forward(const TensorImplPtr& input,
                                             const TensorImplPtr& target,
                                             const TensorImplPtr& weight,
                                             const TensorImplPtr& pos_weight,
                                             Reduction reduction) {
    if (!input || !target || !weight || !pos_weight)
        ErrorBuilder("bce_with_logits").fail("input/target/weight/pos_weight required");
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "bce_with_logits: input/target shape mismatch");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage =
        backend::Dispatcher::for_device(input->device())
            .bce_with_logits_loss(input->storage(), target->storage(), weight->storage(),
                                  pos_weight->storage(), input->shape(), weight->shape(),
                                  pos_weight->shape(), input->dtype(), static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<BCEWithLogitsBackward>();
        bwd->reduction_ = reduction;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<BCEWithLogitsBackward, 4>::wire_autograd(
            std::move(bwd), {input, target, weight, pos_weight}, out);
    }
    return out;
}

std::vector<Storage> BCEWithLogitsBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_).bce_with_logits_backward(
        saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], saved_inputs_[3], grad_out,
        orig_shape_, dtype_, static_cast<int>(reduction_));
}

TensorImplPtr bce_with_logits_op(const TensorImplPtr& input,
                                 const TensorImplPtr& target,
                                 const TensorImplPtr& weight,
                                 const TensorImplPtr& pos_weight,
                                 int reduction) {
    return BCEWithLogitsBackward::forward(input, target, weight, pos_weight,
                                          static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(BCEWithLogitsBackward)

const OpSchema CrossEntropyBackward::schema_v1{"cross_entropy_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr CrossEntropyBackward::forward(const TensorImplPtr& input,
                                            const TensorImplPtr& target,
                                            const TensorImplPtr& weight_or_null,
                                            Reduction reduction,
                                            double eps,
                                            int ignore_index) {
    if (!input || !target)
        ErrorBuilder("cross_entropy").fail("null input");
    if (input->shape().size() < 2)
        throw ShapeMismatch(input->shape(), Shape{}, "cross_entropy: input must be (N, C, ...)");
    if (input->device() != target->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(target->device())),
                             "cross_entropy: input/target");
    if (weight_or_null && weight_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(weight_or_null->device())),
                             "cross_entropy: input/weight");

    const int N = static_cast<int>(input->shape()[0]);

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape().size() > 2) {
        for (std::size_t i = 2; i < input->shape().size(); ++i)
            per_sample_shape.push_back(input->shape()[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(per_sample_shape, reduction)};

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    const Storage* weight_storage = weight_or_null ? &weight_or_null->storage() : nullptr;
    auto result = backend::Dispatcher::for_device(input->device())
                      .cross_entropy_loss(input->storage(), target->storage(), weight_storage,
                                          input->shape(), target->shape(), input->dtype(), eps,
                                          ignore_index, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(result.output), out_shape, input->dtype(),
                                            input->device(), false);

    {
        auto bwd = std::make_shared<CrossEntropyBackward>();
        bwd->reduction_ = reduction;
        bwd->eps_ = eps;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape();
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_softmax_ = std::move(result.saved_aux);
        bwd->saved_target_ = target->storage();
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage();
        bwd->saved_valid_count_ = std::move(result.valid_count);
        kernel::NaryKernel<CrossEntropyBackward, 1>::wire_autograd(std::move(bwd), {input}, out,
                                                                   false);
    }
    return out;
}

std::vector<Storage> CrossEntropyBackward::apply(Storage grad_out) {
    const Storage* weight_storage = has_weight_ ? &saved_weight_ : nullptr;
    auto dx = backend::Dispatcher::for_device(device_).cross_entropy_backward(
        saved_softmax_, saved_target_, weight_storage, saved_valid_count_, grad_out,
        orig_input_shape_, dtype_, ignore_index_, static_cast<int>(reduction_));
    return {std::move(dx)};
}

TensorImplPtr cross_entropy_op(const TensorImplPtr& input,
                               const TensorImplPtr& target,
                               const TensorImplPtr& weight_or_null,
                               int reduction,
                               double eps,
                               int ignore_index) {
    return CrossEntropyBackward::forward(input, target, weight_or_null,
                                         static_cast<Reduction>(reduction), eps, ignore_index);
}
LUCID_REGISTER_OP(CrossEntropyBackward)

const OpSchema NLLLossBackward::schema_v1{"nll_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr NLLLossBackward::forward(const TensorImplPtr& input,
                                       const TensorImplPtr& target,
                                       const TensorImplPtr& weight_or_null,
                                       Reduction reduction,
                                       int ignore_index) {
    if (!input || !target)
        ErrorBuilder("nll_loss").fail("null input");
    if (input->shape().size() < 2)
        throw ShapeMismatch(input->shape(), Shape{}, "nll_loss: input must be (N, C, ...)");
    if (input->device() != target->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(target->device())), "nll_loss: input/target");
    if (weight_or_null && weight_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(weight_or_null->device())),
                             "nll_loss: input/weight");

    const int N = static_cast<int>(input->shape()[0]);

    Shape per_sample_shape;
    per_sample_shape.push_back(static_cast<std::int64_t>(N));
    if (input->shape().size() > 2) {
        for (std::size_t i = 2; i < input->shape().size(); ++i)
            per_sample_shape.push_back(input->shape()[i]);
    }
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(per_sample_shape, reduction)};

    Shape out_shape = (reduction == Reduction::None) ? per_sample_shape : Shape{};
    const Storage* weight_storage = weight_or_null ? &weight_or_null->storage() : nullptr;
    auto result =
        backend::Dispatcher::for_device(input->device())
            .nll_loss(input->storage(), target->storage(), weight_storage, input->shape(),
                      target->shape(), input->dtype(), ignore_index, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(result.output), out_shape, input->dtype(),
                                            input->device(), false);

    {
        auto bwd = std::make_shared<NLLLossBackward>();
        bwd->reduction_ = reduction;
        bwd->ignore_index_ = ignore_index;
        bwd->orig_input_shape_ = input->shape();
        bwd->has_weight_ = (weight_or_null != nullptr);
        bwd->saved_target_ = target->storage();
        if (weight_or_null)
            bwd->saved_weight_ = weight_or_null->storage();
        bwd->saved_valid_count_ = std::move(result.valid_count);
        kernel::NaryKernel<NLLLossBackward, 1>::wire_autograd(std::move(bwd), {input}, out, false);
    }
    return out;
}

std::vector<Storage> NLLLossBackward::apply(Storage grad_out) {
    const Storage* weight_storage = has_weight_ ? &saved_weight_ : nullptr;
    auto dx = backend::Dispatcher::for_device(device_).nll_loss_backward(
        saved_target_, weight_storage, saved_valid_count_, grad_out, orig_input_shape_, dtype_,
        ignore_index_, static_cast<int>(reduction_));
    return {std::move(dx)};
}

TensorImplPtr nll_loss_op(const TensorImplPtr& input,
                          const TensorImplPtr& target,
                          const TensorImplPtr& weight_or_null,
                          int reduction,
                          int ignore_index) {
    return NLLLossBackward::forward(input, target, weight_or_null,
                                    static_cast<Reduction>(reduction), ignore_index);
}
LUCID_REGISTER_OP(NLLLossBackward)

const OpSchema HuberLossBackward::schema_v1{"huber_loss", 1, AmpPolicy::ForceFP32, true};

TensorImplPtr HuberLossBackward::forward(const TensorImplPtr& input,
                                         const TensorImplPtr& target,
                                         double delta,
                                         Reduction reduction) {
    if (!input || !target)
        ErrorBuilder("huber_loss").fail("null input");
    if (input->shape() != target->shape())
        throw ShapeMismatch(input->shape(), target->shape(),
                            "huber_loss: input/target shape mismatch");
    if (delta <= 0.0)
        ErrorBuilder("huber_loss").fail("delta must be positive");

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(),
                      reduced_shape(input->shape(), reduction)};

    Storage out_storage = backend::Dispatcher::for_device(input->device())
                              .huber_loss(input->storage(), target->storage(), input->shape(),
                                          input->dtype(), delta, static_cast<int>(reduction));

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                            reduced_shape(input->shape(), reduction),
                                            input->dtype(), input->device(), false);

    {
        auto bwd = std::make_shared<HuberLossBackward>();
        bwd->reduction_ = reduction;
        bwd->delta_ = delta;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<HuberLossBackward, 2>::wire_autograd(std::move(bwd), {input, target},
                                                                out);
    }
    return out;
}

std::vector<Storage> HuberLossBackward::apply(Storage grad_out) {
    auto grads = backend::Dispatcher::for_device(device_).huber_loss_backward(
        saved_inputs_[0], saved_inputs_[1], grad_out, orig_shape_, dtype_, delta_,
        static_cast<int>(reduction_));
    return {std::move(grads.first), std::move(grads.second)};
}

TensorImplPtr huber_loss_op(const TensorImplPtr& input,
                            const TensorImplPtr& target,
                            double delta,
                            int reduction) {
    return HuberLossBackward::forward(input, target, delta, static_cast<Reduction>(reduction));
}
LUCID_REGISTER_OP(HuberLossBackward)

}  // namespace lucid
