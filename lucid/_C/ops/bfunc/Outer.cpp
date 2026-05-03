// lucid/_C/ops/bfunc/Outer.cpp
//
// Implements OuterBackward and the outer_op free function.
// The outer product is computed via a matrix multiply between reshaped inputs:
//   a[:, None] @ b[None, :]  → [M×1] @ [1×N] = [M×N]

#include "Outer.h"

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
#include "_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Backward node for the outer product C = a ⊗ b.
//
// Given C[i,j] = a[i] * b[j], the gradients are:
//   dA[i] = Σ_j grad_out[i,j] * b[j]  = grad_out @ b (treating b as a column)
//   dB[j] = Σ_i grad_out[i,j] * a[i]  = a^T @ grad_out (treating a as a row)
//
// These are evaluated as matmuls with the saved operands reshaped to 2-D:
//   dA: grad_out [M×N] @ b_col [N×1] → da_col [M×1] → reshape to [M]
//   dB: a_row [1×M] @ grad_out [M×N] → db_row [1×N] → reshape to [N]
class OuterBackward : public AutogradNode<OuterBackward, 2> {
public:
    static const OpSchema schema_v1;

    Storage saved_a_, saved_b_;
    std::int64_t M_, N_;  // Length of a and b respectively.

    // Compute dA and dB by contracting grad_out with the saved operands.
    std::vector<Storage> apply(Storage grad_out) override {
        auto& be = backend::Dispatcher::for_device(device_);

        // Reshape saved vectors to column/row for matmul.
        Shape b_col_shape{N_, 1};
        Shape a_row_shape{1, M_};
        Shape da_col_shape{M_, 1};
        Shape db_row_shape{1, N_};
        Storage b_col = be.reshape(saved_b_, Shape{N_}, b_col_shape, dtype_);
        Storage a_row = be.reshape(saved_a_, Shape{M_}, a_row_shape, dtype_);

        // dA = grad_out [M×N] @ b_col [N×1] → [M×1]
        backend::MatmulOpts da_opts;
        da_opts.M = static_cast<int>(M_);
        da_opts.K = static_cast<int>(N_);
        da_opts.N = 1;

        // dB = a_row [1×M] @ grad_out [M×N] → [1×N]
        backend::MatmulOpts db_opts;
        db_opts.M = 1;
        db_opts.K = static_cast<int>(M_);
        db_opts.N = static_cast<int>(N_);

        Storage da_col = be.matmul(grad_out, b_col, da_opts, dtype_);
        Storage db_row = be.matmul(a_row, grad_out, db_opts, dtype_);

        // Flatten back to 1-D vectors.
        Storage da = be.reshape(da_col, da_col_shape, Shape{M_}, dtype_);
        Storage db = be.reshape(db_row, db_row_shape, Shape{N_}, dtype_);
        return {std::move(da), std::move(db)};
    }
};

const OpSchema OuterBackward::schema_v1{"outer", 1, AmpPolicy::KeepInput, true};

}  // namespace

TensorImplPtr outer_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "outer");
    if (a->shape().size() != 1 || b->shape().size() != 1)
        ErrorBuilder("outer").fail("requires 1-D inputs");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"outer", device, dt, Shape{}};

    // Capture M, N, and saved inputs into the backward node before the output
    // tensor is constructed so that wire_autograd has a fully populated node.
    auto wire_grad = [&](const TensorImplPtr& out) {
        auto bwd = std::make_shared<OuterBackward>();
        bwd->saved_a_ = a->storage();
        bwd->saved_b_ = b->storage();
        bwd->M_ = a->shape()[0];
        bwd->N_ = b->shape()[0];
        kernel::NaryKernel<OuterBackward, 2>::wire_autograd(std::move(bwd), {a, b}, out, false);
    };

    const std::int64_t M = a->shape()[0];
    const std::int64_t N = b->shape()[0];
    Shape out_shape{M, N};
    auto& be = backend::Dispatcher::for_device(device);

    // Reshape vectors to [M×1] and [1×N] so the matmul primitive produces
    // the outer product [M×N] directly.
    Shape a_col_shape{M, 1};
    Shape b_row_shape{1, N};
    Storage a_col = be.reshape(a->storage(), a->shape(), a_col_shape, dt);
    Storage b_row = be.reshape(b->storage(), b->shape(), b_row_shape, dt);
    backend::MatmulOpts opts;
    opts.M = static_cast<int>(M);
    opts.K = 1;
    opts.N = static_cast<int>(N);
    Storage out = be.matmul(a_col, b_row, opts, dt);
    auto t = fresh(std::move(out), out_shape, dt, device);
    wire_grad(t);
    return t;
}

}  // namespace lucid
