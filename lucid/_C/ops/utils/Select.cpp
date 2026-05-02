#include "Select.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/AutogradNode.h"
#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
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

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::numel;
using utils_detail::wrap_axis;

Storage where_branch_storage(const Storage& grad,
                             const Storage& cond,
                             const Shape& shape,
                             Dtype dt,
                             Device device,
                             bool true_branch) {
    return backend::Dispatcher::for_device(device).where_branch(grad, cond, shape, dt, true_branch);
}

Storage gather_backward_storage(const Storage& grad,
                                const Storage& indices,
                                const Shape& input_shape,
                                const Shape& output_shape,
                                int axis,
                                Dtype index_dtype,
                                Dtype dt,
                                Device device) {
    return backend::Dispatcher::for_device(device).gather_backward(
        grad, indices, input_shape, output_shape, axis, index_dtype, dt);
}

Storage diagonal_backward_storage(const Storage& grad,
                                  const Shape& input_shape,
                                  const Shape& output_shape,
                                  int offset,
                                  int axis1,
                                  int axis2,
                                  Dtype dt,
                                  Device device) {
    return backend::Dispatcher::for_device(device).diagonal_backward(
        grad, input_shape, output_shape, offset, axis1, axis2, dt);
}

class WhereBackward : public AutogradNode<WhereBackward, 2> {
public:
    static const OpSchema schema_v1;

    Storage cond_;
    Shape shape_;
    std::weak_ptr<TensorImpl> cond_tensor_;
    std::weak_ptr<TensorImpl> x_tensor_;
    std::weak_ptr<TensorImpl> y_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {where_branch_storage(grad_out, cond_, shape_, dtype_, device_, true),
                where_branch_storage(grad_out, cond_, shape_, dtype_, device_, false)};
    }

    void validate_versions() override {
        check_version_match(cond_tensor_, saved_versions_.size() > 0 ? saved_versions_[0] : 0,
                            schema_v1.name, 0);
        check_version_match(x_tensor_, saved_versions_.size() > 1 ? saved_versions_[1] : 0,
                            schema_v1.name, 1);
        check_version_match(y_tensor_, saved_versions_.size() > 2 ? saved_versions_[2] : 0,
                            schema_v1.name, 2);
    }
};

const OpSchema WhereBackward::schema_v1{"where", 1, AmpPolicy::KeepInput, true};

class MaskedFillBackward : public AutogradNode<MaskedFillBackward, 1> {
public:
    static const OpSchema schema_v1;

    Storage mask_;
    Shape shape_;
    std::weak_ptr<TensorImpl> input_tensor_;
    std::weak_ptr<TensorImpl> mask_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {where_branch_storage(grad_out, mask_, shape_, dtype_, device_, false)};
    }

    void validate_versions() override {
        check_version_match(input_tensor_, saved_versions_.size() > 0 ? saved_versions_[0] : 0,
                            schema_v1.name, 0);
        check_version_match(mask_tensor_, saved_versions_.size() > 1 ? saved_versions_[1] : 0,
                            schema_v1.name, 1);
    }
};

const OpSchema MaskedFillBackward::schema_v1{"masked_fill", 1, AmpPolicy::KeepInput, true};

class RollBackward : public FuncOp<RollBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<std::int64_t> shifts_;
    std::vector<int> axes_;

    std::vector<Storage> apply(Storage grad_out) override {
        std::vector<std::int64_t> inv_shifts;
        inv_shifts.reserve(shifts_.size());
        for (auto s : shifts_)
            inv_shifts.push_back(-s);
        return {backend::Dispatcher::for_device(device_).roll(grad_out, out_shape_, dtype_,
                                                              inv_shifts, axes_)};
    }
};

const OpSchema RollBackward::schema_v1{"roll", 1, AmpPolicy::KeepInput, true};

class GatherBackward : public AutogradNode<GatherBackward, 1> {
public:
    static const OpSchema schema_v1;

    Storage indices_;
    Shape input_shape_;
    Shape output_shape_;
    int axis_ = 0;
    Dtype index_dtype_ = Dtype::I64;
    std::weak_ptr<TensorImpl> input_tensor_;
    std::weak_ptr<TensorImpl> indices_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {gather_backward_storage(grad_out, indices_, input_shape_, output_shape_, axis_,
                                        index_dtype_, dtype_, device_)};
    }

    void validate_versions() override {
        check_version_match(input_tensor_, saved_versions_.size() > 0 ? saved_versions_[0] : 0,
                            schema_v1.name, 0);
        check_version_match(indices_tensor_, saved_versions_.size() > 1 ? saved_versions_[1] : 0,
                            schema_v1.name, 1);
    }
};

const OpSchema GatherBackward::schema_v1{"gather", 1, AmpPolicy::KeepInput, true};

class DiagonalBackward : public FuncOp<DiagonalBackward, 1> {
public:
    static const OpSchema schema_v1;

    int offset_ = 0;
    int axis1_ = 0;
    int axis2_ = 1;

    std::vector<Storage> apply(Storage grad_out) override {
        return {diagonal_backward_storage(grad_out, input_shapes_[0], out_shape_, offset_, axis1_,
                                          axis2_, dtype_, device_)};
    }
};

const OpSchema DiagonalBackward::schema_v1{"diagonal", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_where_grad(const TensorImplPtr& cond,
                                const TensorImplPtr& x,
                                const TensorImplPtr& y,
                                TensorImplPtr out) {
    const bool needs_grad = GradMode::is_enabled() && (x->requires_grad() || y->requires_grad());
    if (!needs_grad)
        return out;

    auto bwd = std::make_shared<WhereBackward>();
    bwd->cond_ = cond->storage();
    bwd->shape_ = out->shape();
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->cond_tensor_ = cond;
    bwd->x_tensor_ = x;
    bwd->y_tensor_ = y;
    bwd->set_next_edges(
        std::vector<Edge>{Edge(detail::ensure_grad_fn(x), 0), Edge(detail::ensure_grad_fn(y), 0)});
    bwd->set_saved_versions({cond->version(), x->version(), y->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr
attach_masked_fill_grad(const TensorImplPtr& a, const TensorImplPtr& mask, TensorImplPtr out) {
    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;

    auto bwd = std::make_shared<MaskedFillBackward>();
    bwd->mask_ = mask->storage();
    bwd->shape_ = out->shape();
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->input_tensor_ = a;
    bwd->mask_tensor_ = mask;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version(), mask->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

template <class Derived>
TensorImplPtr
attach_unary_grad(const TensorImplPtr& a, TensorImplPtr out, std::shared_ptr<Derived> bwd) {
    kernel::NaryKernel<Derived, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(WhereBackward)
LUCID_REGISTER_OP(MaskedFillBackward)
LUCID_REGISTER_OP(RollBackward)
LUCID_REGISTER_OP(GatherBackward)
LUCID_REGISTER_OP(DiagonalBackward)

}  // namespace

TensorImplPtr where_op(const TensorImplPtr& cond, const TensorImplPtr& x, const TensorImplPtr& y) {
    if (!cond || !x || !y)
        ErrorBuilder("where").fail("null input");
    if (x->dtype() != y->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(y->dtype())), "where");
    if (x->device() != y->device() || cond->device() != x->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(y->device())), "where");
    const Dtype dt = x->dtype();
    const Device device = x->device();
    OpScopeFull scope{"where", device, dt, x->shape()};
    if (device == Device::CPU && cond->shape() != x->shape())
        throw ShapeMismatch(x->shape(), y->shape(), "where (CPU same-shape)");
    auto out_storage = backend::Dispatcher::for_device(device).where_op(
        cond->storage(), x->storage(), y->storage(), x->shape(), dt);
    Shape out_shape;
    if (device == Device::GPU) {
        const auto& gs = storage_gpu(out_storage);
        out_shape = mlx_shape_to_lucid(gs.arr->shape());
    } else {
        out_shape = x->shape();
    }
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_where_grad(cond, x, y, std::move(result));
}

TensorImplPtr masked_fill_op(const TensorImplPtr& a, const TensorImplPtr& mask, double value) {
    if (!a || !mask)
        ErrorBuilder("masked_fill").fail("null input");
    if (a->shape() != mask->shape())
        throw ShapeMismatch(a->shape(), mask->shape(), "masked_fill");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"masked_fill", device, dt, a->shape()};
    auto out_storage = backend::Dispatcher::for_device(device).masked_fill(
        a->storage(), mask->storage(), a->shape(), dt, value);
    auto result = fresh(std::move(out_storage), a->shape(), dt, device);
    return attach_masked_fill_grad(a, mask, std::move(result));
}

TensorImplPtr
roll_op(const TensorImplPtr& a, std::vector<std::int64_t> shifts, std::vector<int> axes) {
    Validator::input(a, "roll.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"roll", device, dt, a->shape()};
    if (shifts.size() != axes.size())
        ErrorBuilder("roll").fail("shifts and axes must have equal length");
    const std::size_t ndim = a->shape().size();
    for (std::size_t i = 0; i < axes.size(); ++i)
        axes[i] = wrap_axis(axes[i], static_cast<int>(ndim));
    auto out_storage =
        backend::Dispatcher::for_device(device).roll(a->storage(), a->shape(), dt, shifts, axes);
    auto result = fresh(std::move(out_storage), a->shape(), dt, device);
    auto bwd = std::make_shared<RollBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->shifts_ = std::move(shifts);
    bwd->axes_ = std::move(axes);
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

TensorImplPtr gather_op(const TensorImplPtr& a, const TensorImplPtr& indices, int axis) {
    if (!a || !indices)
        ErrorBuilder("gather").fail("null input");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"gather", device, dt, indices->shape()};
    if (a->shape().size() != indices->shape().size())
        throw ShapeMismatch(a->shape(), indices->shape(),
                            "gather: a and indices must have same rank");
    const std::size_t ndim = a->shape().size();
    int ax = wrap_axis(axis, static_cast<int>(ndim));
    Shape out_shape = indices->shape();
    auto out_storage = backend::Dispatcher::for_device(device).gather(
        a->storage(), indices->storage(), a->shape(), out_shape, ax, indices->dtype(), dt);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    if (GradMode::is_enabled() && a->requires_grad()) {
        auto bwd = std::make_shared<GatherBackward>();
        bwd->indices_ = indices->storage();
        bwd->input_shape_ = a->shape();
        bwd->output_shape_ = result->shape();
        bwd->axis_ = ax;
        bwd->dtype_ = dt;
        bwd->index_dtype_ = indices->dtype();
        bwd->device_ = device;
        bwd->input_tensor_ = a;
        bwd->indices_tensor_ = indices;
        bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
        bwd->set_saved_versions({a->version(), indices->version()});
        result->set_grad_fn(std::move(bwd));
        result->set_leaf(false);
        result->set_requires_grad(true);
    }
    return result;
}

TensorImplPtr diagonal_op(const TensorImplPtr& a, int offset, int axis1, int axis2) {
    Validator::input(a, "diagonal.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"diagonal", device, dt, a->shape()};
    const std::size_t ndim = a->shape().size();
    if (ndim < 2)
        ErrorBuilder("diagonal").fail("input must be ≥2-D");
    int a1 = wrap_axis(axis1, static_cast<int>(ndim));
    int a2 = wrap_axis(axis2, static_cast<int>(ndim));
    if (a1 == a2)
        ErrorBuilder("diagonal").fail("axis1 and axis2 must differ");
    if (a1 > a2)
        std::swap(a1, a2);

    const std::int64_t M = a->shape()[a1];
    const std::int64_t N = a->shape()[a2];
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

    Shape out_shape;
    for (std::size_t d = 0; d < ndim; ++d) {
        if ((int)d == a1 || (int)d == a2)
            continue;
        out_shape.push_back(a->shape()[d]);
    }
    out_shape.push_back(L);
    auto out_storage = backend::Dispatcher::for_device(device).diagonal(a->storage(), a->shape(),
                                                                        offset, a1, a2, dt);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    auto bwd = std::make_shared<DiagonalBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->offset_ = offset;
    bwd->axis1_ = a1;
    bwd->axis2_ = a2;
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

}  // namespace lucid
