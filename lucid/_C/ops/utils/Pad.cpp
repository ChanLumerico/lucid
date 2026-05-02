#include "Pad.h"

#include <variant>

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::fresh;

class PadBackward : public FuncOp<PadBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<std::pair<std::int64_t, std::int64_t>> pad_width_;

    std::vector<Storage> apply(Storage grad_out) override {
        auto& be = backend::Dispatcher::for_device(device_);
        Storage current = std::move(grad_out);
        Shape current_shape = out_shape_;
        for (std::size_t d = 0; d < input_shapes_[0].size(); ++d) {
            Shape next_shape = current_shape;
            next_shape[d] = input_shapes_[0][d];
            current = be.slice_axis(current, current_shape, next_shape, static_cast<int>(d),
                                    pad_width_[d].first, dtype_);
            current_shape = std::move(next_shape);
        }
        return {std::move(current)};
    }
};

const OpSchema PadBackward::schema_v1{"pad", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_pad_grad(const TensorImplPtr& a,
                              TensorImplPtr out,
                              std::vector<std::pair<std::int64_t, std::int64_t>> pad_width) {
    auto bwd = std::make_shared<PadBackward>();
    bwd->pad_width_ = std::move(pad_width);
    kernel::NaryKernel<PadBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(PadBackward)

}  // namespace

TensorImplPtr pad_op(const TensorImplPtr& a,
                     std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
                     double constant) {
    Validator::input(a, "pad.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"pad", device, dt, a->shape()};
    if (pad_width.size() != a->shape().size())
        ErrorBuilder("pad").fail("pad_width length must equal ndim");
    const std::size_t ndim = a->shape().size();
    Shape out_shape(ndim);
    for (std::size_t d = 0; d < ndim; ++d)
        out_shape[d] = a->shape()[d] + pad_width[d].first + pad_width[d].second;

    Storage out_storage = backend::Dispatcher::for_device(device).pad(a->storage(), a->shape(), dt,
                                                                      pad_width, constant);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_pad_grad(a, std::move(result), std::move(pad_width));
}

}  // namespace lucid
