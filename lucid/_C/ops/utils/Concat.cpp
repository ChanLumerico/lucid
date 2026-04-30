#include "Concat.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

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
#include "../../kernel/VariadicKernel.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::check_dtype_device_match;
using utils_detail::fresh;
using utils_detail::wrap_axis;

Storage slice_axis_storage(const Storage& src,
                           const Shape& src_shape,
                           const Shape& slice_shape,
                           int axis,
                           std::int64_t offset,
                           Dtype dt,
                           Device device) {
    return backend::Dispatcher::for_device(device).slice_axis(src, src_shape, slice_shape, axis,
                                                              offset, dt);
}

Storage insert_axis_slice_storage(const Storage& src,
                                  const Shape& src_shape,
                                  const Shape& dst_shape,
                                  int axis,
                                  std::int64_t offset,
                                  Dtype dt,
                                  Device device) {
    return backend::Dispatcher::for_device(device).insert_axis_slice(src, src_shape, dst_shape,
                                                                     axis, offset, dt);
}

class ConcatBackward : public kernel::VariadicKernel<ConcatBackward> {
public:
    static const OpSchema schema_v1;

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_;
    std::vector<Shape> input_shapes_;
    Shape output_shape_;
    int axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        std::vector<Storage> grads;
        grads.reserve(input_shapes_.size());
        std::int64_t offset = 0;
        for (const auto& shape : input_shapes_) {
            grads.push_back(
                slice_axis_storage(grad_out, output_shape_, shape, axis_, offset, dtype_, device_));
            offset += shape[static_cast<std::size_t>(axis_)];
        }
        return grads;
    }

    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            check_version_match(input_tensors_[i],
                                saved_versions_.size() > i ? saved_versions_[i] : 0, schema_v1.name,
                                i);
        }
    }
};

const OpSchema ConcatBackward::schema_v1{"concatenate", 1, AmpPolicy::KeepInput, true};

class StackBackward : public kernel::VariadicKernel<StackBackward> {
public:
    static const OpSchema schema_v1;

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_;
    Shape input_shape_;
    Shape output_shape_;
    int axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        Shape slice_shape = output_shape_;
        slice_shape[static_cast<std::size_t>(axis_)] = 1;
        std::vector<Storage> grads;
        grads.reserve(input_tensors_.size());
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            auto piece = slice_axis_storage(grad_out, output_shape_, slice_shape, axis_,
                                            static_cast<std::int64_t>(i), dtype_, device_);
            grads.push_back(backend::Dispatcher::for_device(device_).reshape(
                piece, slice_shape, input_shape_, dtype_));
        }
        return grads;
    }

    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            check_version_match(input_tensors_[i],
                                saved_versions_.size() > i ? saved_versions_[i] : 0, schema_v1.name,
                                i);
        }
    }
};

const OpSchema StackBackward::schema_v1{"stack", 1, AmpPolicy::KeepInput, true};

class SplitSliceBackward : public FuncOp<SplitSliceBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape slice_shape_;
    int axis_ = 0;
    std::int64_t offset_ = 0;
    bool squeeze_axis_ = false;

    std::vector<Storage> apply(Storage grad_out) override {
        Storage slice_grad = std::move(grad_out);
        if (squeeze_axis_) {
            slice_grad = backend::Dispatcher::for_device(device_).reshape(
                slice_grad, out_shape_, slice_shape_, dtype_);
        }
        return {insert_axis_slice_storage(slice_grad, slice_shape_, input_shapes_[0], axis_,
                                          offset_, dtype_, device_)};
    }
};

const OpSchema SplitSliceBackward::schema_v1{"split", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_concat_grad(const std::vector<TensorImplPtr>& xs,
                                 TensorImplPtr out,
                                 int axis) {
    bool needs_grad = GradMode::is_enabled();
    if (needs_grad) {
        needs_grad = false;
        for (const auto& t : xs)
            needs_grad = needs_grad || t->requires_grad();
    }
    if (!needs_grad)
        return out;

    auto bwd = std::make_shared<ConcatBackward>();
    bwd->input_shapes_.reserve(xs.size());
    bwd->input_tensors_.reserve(xs.size());
    bwd->output_shape_ = out->shape();
    bwd->axis_ = axis;
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();

    std::vector<Edge> edges;
    std::vector<std::int64_t> versions;
    edges.reserve(xs.size());
    versions.reserve(xs.size());
    for (const auto& t : xs) {
        bwd->input_shapes_.push_back(t->shape());
        bwd->input_tensors_.push_back(t);
        edges.emplace_back(detail::ensure_grad_fn(t), 0);
        versions.push_back(t->version());
    }
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions(std::move(versions));

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr attach_stack_grad(const std::vector<TensorImplPtr>& xs, TensorImplPtr out, int axis) {
    bool needs_grad = GradMode::is_enabled();
    if (needs_grad) {
        needs_grad = false;
        for (const auto& t : xs)
            needs_grad = needs_grad || t->requires_grad();
    }
    if (!needs_grad)
        return out;

    auto bwd = std::make_shared<StackBackward>();
    bwd->input_shape_ = xs[0]->shape();
    bwd->output_shape_ = out->shape();
    bwd->axis_ = axis;
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->input_tensors_.reserve(xs.size());

    std::vector<Edge> edges;
    std::vector<std::int64_t> versions;
    edges.reserve(xs.size());
    versions.reserve(xs.size());
    for (const auto& t : xs) {
        bwd->input_tensors_.push_back(t);
        edges.emplace_back(detail::ensure_grad_fn(t), 0);
        versions.push_back(t->version());
    }
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions(std::move(versions));

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr attach_split_grad(const TensorImplPtr& a,
                                TensorImplPtr out,
                                Shape slice_shape,
                                int axis,
                                std::int64_t offset,
                                bool squeeze_axis) {
    auto bwd = std::make_shared<SplitSliceBackward>();
    bwd->slice_shape_ = std::move(slice_shape);
    bwd->axis_ = axis;
    bwd->offset_ = offset;
    bwd->squeeze_axis_ = squeeze_axis;
    kernel::NaryKernel<SplitSliceBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                             /*save_ins=*/false);
    return out;
}

LUCID_REGISTER_OP(ConcatBackward)
LUCID_REGISTER_OP(StackBackward)
LUCID_REGISTER_OP(SplitSliceBackward)

}  // namespace

TensorImplPtr concatenate_op(const std::vector<TensorImplPtr>& xs, int axis) {
    check_dtype_device_match(xs, "concatenate");
    const Dtype dt = xs[0]->dtype();
    const Device device = xs[0]->device();
    OpScopeFull scope{"concatenate", device, dt, Shape{}};
    const auto ndim = xs[0]->shape().size();
    int ax = wrap_axis(axis, static_cast<int>(ndim));

    Shape out_shape = xs[0]->shape();
    std::int64_t cat_dim = 0;
    std::vector<Storage> storages;
    std::vector<Shape> shapes;
    storages.reserve(xs.size());
    shapes.reserve(xs.size());
    for (auto& t : xs) {
        if (t->shape().size() != ndim)
            throw ShapeMismatch(xs[0]->shape(), t->shape(), "concatenate");
        for (std::size_t d = 0; d < ndim; ++d) {
            if ((int)d == ax)
                continue;
            if (t->shape()[d] != xs[0]->shape()[d])
                throw ShapeMismatch(xs[0]->shape(), t->shape(), "concatenate");
        }
        cat_dim += t->shape()[ax];
        storages.push_back(t->storage());
        shapes.push_back(t->shape());
    }
    out_shape[ax] = cat_dim;
    auto out_storage =
        backend::Dispatcher::for_device(device).concatenate(storages, shapes, ax, dt);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_concat_grad(xs, std::move(result), ax);
}

TensorImplPtr stack_op(const std::vector<TensorImplPtr>& xs, int axis) {
    check_dtype_device_match(xs, "stack");
    const Dtype dt = xs[0]->dtype();
    const Device device = xs[0]->device();
    OpScopeFull scope{"stack", device, dt, Shape{}};
    const auto ndim = xs[0]->shape().size();
    int ax = axis < 0 ? axis + static_cast<int>(ndim) + 1 : axis;
    if (ax < 0 || ax > static_cast<int>(ndim)) {
        ErrorBuilder("stack").index_error("axis out of range");
    }
    for (const auto& t : xs) {
        if (t->shape() != xs[0]->shape()) {
            throw ShapeMismatch(xs[0]->shape(), t->shape(), "stack");
        }
    }

    Shape out_shape = xs[0]->shape();
    out_shape.insert(out_shape.begin() + ax, static_cast<std::int64_t>(xs.size()));
    std::vector<Storage> storages;
    storages.reserve(xs.size());
    for (const auto& t : xs)
        storages.push_back(t->storage());
    auto out_storage =
        backend::Dispatcher::for_device(device).stack(storages, xs[0]->shape(), ax, dt);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_stack_grad(xs, std::move(result), ax);
}

TensorImplPtr hstack_op(const std::vector<TensorImplPtr>& xs) {
    if (xs.empty())
        ErrorBuilder("hstack").fail("empty input");
    return concatenate_op(xs, xs[0]->shape().size() <= 1 ? 0 : 1);
}

TensorImplPtr vstack_op(const std::vector<TensorImplPtr>& xs) {
    if (xs.empty())
        ErrorBuilder("vstack").fail("empty input");
    if (xs[0]->shape().size() == 1) {
        return stack_op(xs, 0);
    }
    return concatenate_op(xs, 0);
}

std::vector<TensorImplPtr> split_op(const TensorImplPtr& a, std::int64_t num_splits, int axis) {
    Validator::input(a, "split.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"split", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    if (num_splits <= 0)
        ErrorBuilder("split").fail("num_splits must be positive");
    if (a->shape()[ax] % num_splits != 0)
        ErrorBuilder("split").fail("dimension not divisible by num_splits");
    const std::int64_t piece = a->shape()[ax] / num_splits;
    Shape piece_shape = a->shape();
    piece_shape[ax] = piece;
    auto pieces =
        backend::Dispatcher::for_device(device).split_equal(a->storage(), a->shape(), ax,
                                                            num_splits, dt);
    std::vector<TensorImplPtr> out;
    out.reserve(pieces.size());
    std::int64_t off = 0;
    for (auto& piece_storage : pieces) {
        auto result = fresh(std::move(piece_storage), piece_shape, dt, device);
        out.push_back(attach_split_grad(a, std::move(result), piece_shape, ax, off,
                                        /*squeeze_axis=*/false));
        off += piece;
    }
    return out;
}

std::vector<TensorImplPtr> split_at_op(const TensorImplPtr& a,
                                       std::vector<std::int64_t> indices,
                                       int axis) {
    Validator::input(a, "split_at.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"split_at", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    auto pieces =
        backend::Dispatcher::for_device(device).split_at(a->storage(), a->shape(), ax, indices, dt);
    std::vector<TensorImplPtr> out;
    out.reserve(pieces.size());
    std::int64_t off = 0;
    std::size_t piece_idx = 0;
    for (auto& piece_storage : pieces) {
        Shape piece_shape = a->shape();
        const std::int64_t next =
            (piece_idx < indices.size()) ? indices[piece_idx] : a->shape()[ax];
        piece_shape[ax] = next - off;
        auto result = fresh(std::move(piece_storage), piece_shape, dt, device);
        out.push_back(
            attach_split_grad(a, std::move(result), piece_shape, ax, off, /*squeeze_axis=*/false));
        off = next;
        ++piece_idx;
    }
    return out;
}

std::vector<TensorImplPtr> chunk_op(const TensorImplPtr& a, std::int64_t chunks, int axis) {
    return split_op(a, chunks, axis);
}

std::vector<TensorImplPtr> unbind_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "unbind.a").non_null();
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    std::vector<TensorImplPtr> out;
    out.reserve(static_cast<std::size_t>(a->shape()[ax]));
    Shape slice_shape = a->shape();
    slice_shape[static_cast<std::size_t>(ax)] = 1;
    Shape out_shape = slice_shape;
    out_shape.erase(out_shape.begin() + ax);

    for (std::int64_t k = 0; k < a->shape()[ax]; ++k) {
        auto piece = slice_axis_storage(a->storage(), a->shape(), slice_shape, ax, k, a->dtype(),
                                        a->device());
        auto reshaped = backend::Dispatcher::for_device(a->device()).reshape(
            piece, slice_shape, out_shape, a->dtype());
        auto result = fresh(std::move(reshaped), out_shape, a->dtype(), a->device());
        out.push_back(
            attach_split_grad(a, std::move(result), slice_shape, ax, k, /*squeeze_axis=*/true));
    }
    return out;
}

}  // namespace lucid
