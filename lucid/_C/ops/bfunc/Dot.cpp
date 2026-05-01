#include "Dot.h"

#include <variant>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Backward for 1-D × 1-D dot: da = b * grad, db = a * grad (grad is scalar).
class Dot1DBackward : public AutogradNode<Dot1DBackward, 2> {
public:
    static const OpSchema schema_v1;

    Storage saved_a_, saved_b_;
    std::size_t numel_;

    std::vector<Storage> apply(Storage grad_out) override {
        Shape vec_shape{static_cast<std::int64_t>(numel_)};
        auto& be = backend::Dispatcher::for_device(device_);
        Storage scaled_grad = be.broadcast(grad_out, Shape{}, vec_shape, dtype_);
        Storage da = be.mul(saved_b_, scaled_grad, vec_shape, dtype_);
        Storage db = be.mul(saved_a_, scaled_grad, vec_shape, dtype_);
        return {std::move(da), std::move(db)};
    }
};

// Backward for 2-D × 2-D dot: da = grad @ b.T, db = a.T @ grad.
class Dot2DBackward : public AutogradNode<Dot2DBackward, 2> {
public:
    static const OpSchema schema_v1;

    Storage saved_a_, saved_b_;
    Shape a_shape_, b_shape_;

    std::vector<Storage> apply(Storage grad_out) override {
        const std::int64_t M = a_shape_[0], K = a_shape_[1], N = b_shape_[1];
        backend::MatmulOpts da_opts;
        da_opts.M = static_cast<int>(M);
        da_opts.K = static_cast<int>(N);
        da_opts.N = static_cast<int>(K);
        da_opts.transB = true;
        backend::MatmulOpts db_opts;
        db_opts.M = static_cast<int>(K);
        db_opts.K = static_cast<int>(M);
        db_opts.N = static_cast<int>(N);
        db_opts.transA = true;
        auto& be = backend::Dispatcher::for_device(device_);
        Storage da = be.matmul(grad_out, saved_b_, da_opts, dtype_);
        Storage db = be.matmul(saved_a_, grad_out, db_opts, dtype_);
        return {std::move(da), std::move(db)};
    }
};

const OpSchema Dot1DBackward::schema_v1{"dot_1d", 1, AmpPolicy::KeepInput, true};
const OpSchema Dot2DBackward::schema_v1{"dot_2d", 1, AmpPolicy::KeepInput, true};

}  // namespace

TensorImplPtr dot_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "dot");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"dot", device, dt, Shape{}};

    auto wire_grad = [&](const TensorImplPtr& out) {
        if (a->shape().size() == 1 && b->shape().size() == 1) {
            auto n = std::make_shared<Dot1DBackward>();
            n->saved_a_ = a->storage();
            n->saved_b_ = b->storage();
            n->numel_ = static_cast<std::size_t>(a->shape()[0]);
            kernel::NaryKernel<Dot1DBackward, 2>::wire_autograd(std::move(n), {a, b}, out,
                                                                /*save_ins=*/false);
        } else if (a->shape().size() == 2 && b->shape().size() == 2) {
            auto n = std::make_shared<Dot2DBackward>();
            n->saved_a_ = a->storage();
            n->saved_b_ = b->storage();
            n->a_shape_ = a->shape();
            n->b_shape_ = b->shape();
            kernel::NaryKernel<Dot2DBackward, 2>::wire_autograd(std::move(n), {a, b}, out,
                                                                /*save_ins=*/false);
        }
    };

    if (a->shape().size() == 1 && b->shape().size() == 1) {
        if (a->shape()[0] != b->shape()[0])
            throw ShapeMismatch(a->shape(), b->shape(), "dot");
        Shape out_shape{};
        auto& be = backend::Dispatcher::for_device(device);
        Storage values = be.mul(a->storage(), b->storage(), a->shape(), dt);
        backend::ReduceOpts opts{{0}, false};
        Storage out = be.reduce_sum(values, a->shape(), opts, dt);
        auto t = fresh(std::move(out), out_shape, dt, device);
        wire_grad(t);
        return t;
    }

    if (a->shape().size() == 2 && b->shape().size() == 2) {
        const std::int64_t M = a->shape()[0], K = a->shape()[1];
        const std::int64_t Kb = b->shape()[0], N = b->shape()[1];
        if (K != Kb)
            throw ShapeMismatch(a->shape(), b->shape(), "dot");
        Shape out_shape{M, N};
        backend::MatmulOpts opts;
        opts.M = static_cast<int>(M);
        opts.K = static_cast<int>(K);
        opts.N = static_cast<int>(N);
        Storage out =
            backend::Dispatcher::for_device(device).matmul(a->storage(), b->storage(), opts, dt);
        auto t = fresh(std::move(out), out_shape, dt, device);
        wire_grad(t);
        return t;
    }

    ErrorBuilder("dot").not_implemented("CPU supports only 1-D × 1-D and 2-D × 2-D");
}

}  // namespace lucid
