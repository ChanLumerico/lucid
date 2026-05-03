// lucid/_C/ops/utils/Concat.cpp
//
// Implements concatenation, stacking, splitting, chunking, and unbinding.
// Three private autograd nodes live here:
//
//   ConcatBackward      — slices the output gradient into per-input pieces
//                         using the recorded per-input sizes along `axis_`.
//   StackBackward       — slices out each size-1 slab along `axis_` and
//                         reshapes it to drop the stacked dimension.
//   SplitSliceBackward  — zero-pads a split/unbind slice gradient back into
//                         a buffer of the full input shape using offset+axis.
//
// ConcatBackward and StackBackward use VariadicKernel because they own a
// variable-length set of edges and must run validate_versions over each of
// them.  SplitSliceBackward uses FuncOp<N=1> because each split piece owns a
// single edge back to the unsplit input.

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
#include "../bfunc/_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::check_dtype_device_match;
using utils_detail::fresh;
using utils_detail::wrap_axis;

// Extract a contiguous sub-tensor from `src` along `axis` starting at element
// `offset`, with the destination shape given by `slice_shape`.  The slice
// width along `axis` is taken from `slice_shape[axis]`.
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

// Write `src` into a zero-initialised buffer of `dst_shape`, placing the
// slice contents starting at `offset` along `axis`.  All elements outside the
// placed region remain zero.  Used by SplitSliceBackward to reconstruct the
// full input gradient from a per-split-piece gradient.
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

// Backward node for concatenate_op.
//
// Invariants:
//   input_shapes_  — shapes of every input tensor in the order they were
//                    concatenated.  input_shapes_[i][axis_] is the width of
//                    the i-th input's contribution along the concat axis.
//   output_shape_  — the shape of the concatenated output tensor.
//   axis_          — the dimension along which concatenation was performed.
//   input_tensors_ — weak references for version-check validation.
//
// Backward formula: the output gradient is split back into per-input pieces
// by slicing at the cumulative boundary offsets along axis_.  The offset
// advances by shape[axis_] after each slice so that each piece corresponds
// exactly to the region that input i wrote into the forward output.
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
            // Advance the slice window by this input's contribution along axis_.
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

// Backward node for stack_op.
//
// Invariants:
//   input_shape_   — the common shape of every stacked input (all inputs had
//                    identical shapes in the forward pass).
//   output_shape_  — the shape of the stacked output (input_shape_ with a
//                    new axis of size N inserted at axis_).
//   axis_          — the axis that was inserted by stack_op.
//   input_tensors_ — weak references for version-check validation.
//
// Backward formula: each input i occupies exactly one slab of size 1 at
// offset i along axis_.  Extract that slab with slice_axis_storage, then
// reshape it from (... 1 ...) to input_shape_ (dropping the size-1 axis).
class StackBackward : public kernel::VariadicKernel<StackBackward> {
public:
    static const OpSchema schema_v1;

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_;
    Shape input_shape_;
    Shape output_shape_;
    int axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        // slice_shape is the output shape with axis_ set to 1, used to
        // select the single slab belonging to each input.
        Shape slice_shape = output_shape_;
        slice_shape[static_cast<std::size_t>(axis_)] = 1;
        std::vector<Storage> grads;
        grads.reserve(input_tensors_.size());
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            // Extract the single-element slab for input i, then drop axis_.
            auto piece = slice_axis_storage(grad_out, output_shape_, slice_shape, axis_,
                                            static_cast<std::int64_t>(i), dtype_, device_);
            // Reshape (drop the size-1 stacked dimension) to recover input_shape_.
            grads.push_back(backend::Dispatcher::for_device(device_).reshape(piece, slice_shape,
                                                                             input_shape_, dtype_));
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

// Backward node for a single split or unbind output slice.
//
// Invariants:
//   slice_shape_  — the shape of the slice in the forward output (includes
//                   the split axis with its piece size, i.e. size 1 for unbind).
//   axis_         — the axis along which splitting was performed.
//   offset_       — element offset of this piece along axis_ in the original
//                   input tensor.
//   squeeze_axis_ — true for unbind: the forward call reshaped the size-1 slab
//                   to drop axis_, so the backward must first un-squeeze to
//                   (... 1 ...) before calling insert_axis_slice.
//
// Backward formula: place the received gradient at `offset_` inside a
// zero-initialised buffer of input_shapes_[0] using insert_axis_slice_storage.
// For unbind (squeeze_axis_=true) the gradient is first reshaped from
// out_shape_ back to slice_shape_ to restore the missing axis before scatter.
class SplitSliceBackward : public FuncOp<SplitSliceBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape slice_shape_;
    int axis_ = 0;
    std::int64_t offset_ = 0;
    // True for unbind: the forward dropped axis_ via reshape, so the backward
    // must restore it before calling insert_axis_slice.
    bool squeeze_axis_ = false;

    std::vector<Storage> apply(Storage grad_out) override {
        Storage slice_grad = std::move(grad_out);
        if (squeeze_axis_) {
            // Re-insert the size-1 axis so that insert_axis_slice_storage
            // can correctly address the offset within the original tensor.
            slice_grad = backend::Dispatcher::for_device(device_).reshape(slice_grad, out_shape_,
                                                                          slice_shape_, dtype_);
        }
        return {insert_axis_slice_storage(slice_grad, slice_shape_, input_shapes_[0], axis_,
                                          offset_, dtype_, device_)};
    }
};

const OpSchema SplitSliceBackward::schema_v1{"split", 1, AmpPolicy::KeepInput, true};

// Build and attach a ConcatBackward node to `out` if gradient mode is on and
// at least one input tensor requires a gradient.  Records each input's shape
// and a weak reference for later version-mismatch detection.
//
// `axis` must already be the normalised (non-negative) axis value; the caller
// is responsible for calling wrap_axis before invoking this helper.
TensorImplPtr
attach_concat_grad(const std::vector<TensorImplPtr>& xs, TensorImplPtr out, int axis) {
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

// Build and attach a StackBackward node to `out` if gradient mode is on and
// at least one input tensor requires a gradient.  Because all inputs to
// stack_op share the same shape, a single input_shape_ field suffices.
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

// Wire a SplitSliceBackward node onto the split/unbind output tensor `out`.
//
// `slice_shape` — the shape before any squeeze (i.e. the slab shape that
//                 includes the axis dimension; its axis entry is the piece
//                 width for split, or 1 for unbind).
// `offset`      — element start position along `axis` in the original tensor.
// `squeeze_axis`— true when called from unbind_op; instructs the backward to
//                 re-insert the dropped axis before scatter.
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
    kernel::NaryKernel<SplitSliceBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(ConcatBackward)
LUCID_REGISTER_OP(StackBackward)
LUCID_REGISTER_OP(SplitSliceBackward)

}  // namespace

// Validate that all inputs share dtype, device, and rank.  Compute the output
// shape by summing each input's extent along `axis` (all other dimensions must
// match).  Delegate the physical memory concatenation to the backend dispatcher,
// then attach ConcatBackward to route gradients back to each input.
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

// Validate that all inputs share dtype, device, and identical shape.  Insert
// a new axis of size xs.size() at position `ax` in the output shape, delegate
// the physical stacking to the backend, then attach StackBackward.
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

// hstack: concatenate along axis 1 for >=2-D inputs (column-wise), or
// along axis 0 for 1-D inputs (element-wise) to match NumPy/PyTorch semantics.
TensorImplPtr hstack_op(const std::vector<TensorImplPtr>& xs) {
    if (xs.empty())
        ErrorBuilder("hstack").fail("empty input");
    return concatenate_op(xs, xs[0]->shape().size() <= 1 ? 0 : 1);
}

// vstack: concatenate along axis 0 for >=2-D inputs (row-wise), but for 1-D
// inputs call stack_op(xs, 0) to produce a 2-D matrix where each input
// becomes a row.  This mirrors the NumPy/PyTorch vstack convention.
TensorImplPtr vstack_op(const std::vector<TensorImplPtr>& xs) {
    if (xs.empty())
        ErrorBuilder("vstack").fail("empty input");
    if (xs[0]->shape().size() == 1) {
        return stack_op(xs, 0);
    }
    return concatenate_op(xs, 0);
}

// Divide `a` into `num_splits` equal-sized pieces along `axis`.  The size of
// `a` along `axis` must be evenly divisible by `num_splits`.  Each piece
// receives an independent SplitSliceBackward node recording its starting
// offset so that gradients can be scattered back to the correct region.
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
    auto pieces = backend::Dispatcher::for_device(device).split_equal(a->storage(), a->shape(), ax,
                                                                      num_splits, dt);
    std::vector<TensorImplPtr> out;
    out.reserve(pieces.size());
    std::int64_t off = 0;
    for (auto& piece_storage : pieces) {
        auto result = fresh(std::move(piece_storage), piece_shape, dt, device);
        out.push_back(attach_split_grad(a, std::move(result), piece_shape, ax, off, false));
        off += piece;
    }
    return out;
}

// Split `a` at explicit cut-point indices along `axis`.  Produces
// (indices.size() + 1) pieces with potentially unequal sizes.  The i-th
// piece spans [indices[i-1], indices[i]) along `axis`, with 0 as the implicit
// left boundary and shape[ax] as the implicit right boundary.  Each piece
// gets a SplitSliceBackward node that records its offset for the backward pass.
std::vector<TensorImplPtr>
split_at_op(const TensorImplPtr& a, std::vector<std::int64_t> indices, int axis) {
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
        out.push_back(attach_split_grad(a, std::move(result), piece_shape, ax, off, false));
        off = next;
        ++piece_idx;
    }
    return out;
}

// chunk_op is a thin alias for split_op; `chunks` is forwarded directly as
// `num_splits`.  The same divisibility requirement applies.
std::vector<TensorImplPtr> chunk_op(const TensorImplPtr& a, std::int64_t chunks, int axis) {
    return split_op(a, chunks, axis);
}

// Decompose `a` along `axis` into individual rank-(ndim-1) slices with the
// axis dimension removed.  For each k in [0, shape[ax]), the k-th output is
// the slice at index k with the axis dimension squeezed out.
//
// The backward uses squeeze_axis_=true so that SplitSliceBackward first
// re-inserts the dropped axis (reshaping from rank ndim-1 back to the
// size-1 slab shape) before scattering the gradient back.
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
        auto reshaped = backend::Dispatcher::for_device(a->device())
                            .reshape(piece, slice_shape, out_shape, a->dtype());
        auto result = fresh(std::move(reshaped), out_shape, a->dtype(), a->device());
        out.push_back(attach_split_grad(a, std::move(result), slice_shape, ax, k, true));
    }
    return out;
}

}  // namespace lucid
