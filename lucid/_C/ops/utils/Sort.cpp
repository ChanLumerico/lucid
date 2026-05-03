// lucid/_C/ops/utils/Sort.cpp
//
// Implements sort, argsort, argmax, argmin, nonzero, unique, and topk.
//
// sort_op and the topk values output are differentiable for F32/F64 inputs.
// Their shared autograd node is IndexScatterBackward, which uses
// Dispatcher::scatter_add_axis to place each incoming gradient value back at
// the position in the input that produced the corresponding sorted output
// element.  This is the adjoint of the gather operation performed during sort.
//
// argsort, argmax, argmin return integer index tensors and have no gradient.
//
// nonzero and unique always execute on CPU because their output sizes are
// determined by data values and cannot be known statically; GPU tensors are
// transferred to CPU before computation.

#include "Sort.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "../../autograd/FuncOp.h"
#include "../../backend/Dispatcher.h"
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

using utils_detail::fresh;
using utils_detail::wrap_axis;

// Integer dtypes cannot accumulate meaningful gradients through sort (the
// sorted positions are discontinuous).  Only F32 and F64 are considered
// differentiable here.
bool differentiable_dtype(Dtype dt) {
    return dt == Dtype::F32 || dt == Dtype::F64;
}

// Scatter-add gradient values from `grad` back to their original positions
// in a zero-initialised buffer of `input_shape`.  Element `grad[i]` at
// position i along `axis` is accumulated into position `indices[i]` in the
// output buffer.  Multiple source positions can map to the same destination
// (scatter-add semantics); this is handled by the backend.
Storage scatter_axis_add_storage(const Storage& grad,
                                 const Storage& indices,
                                 const Shape& input_shape,
                                 const Shape& grad_shape,
                                 int axis,
                                 Dtype dt,
                                 Device device) {
    return backend::Dispatcher::for_device(device).scatter_add_axis(grad, indices, input_shape,
                                                                    grad_shape, axis, dt);
}

// Backward node for sort_op and the values output of topk_op.
//
// Invariants:
//   indices_ — the sort permutation produced by the forward pass.  Element
//              indices_[i] along `axis_` holds the original position of the
//              element that ended up at position i in the sorted output.
//   grad_shape_ — shape of the sorted output (and of grad_out).
//   axis_       — the axis along which sorting was done.
//
// Backward formula: scatter-add grad_out to input positions using indices_,
// which exactly inverts the gather performed during the forward sort.
class IndexScatterBackward : public FuncOp<IndexScatterBackward, 1> {
public:
    static const OpSchema schema_v1;

    Storage indices_;
    Shape grad_shape_;
    int axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        return {scatter_axis_add_storage(grad_out, indices_, input_shapes_[0], grad_shape_, axis_,
                                         dtype_, device_)};
    }
};

const OpSchema IndexScatterBackward::schema_v1{
    "index_scatter", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

// Wire IndexScatterBackward onto the sorted or topk output tensor.  Skips
// wiring entirely for non-differentiable dtypes (integer types) to avoid
// allocating a backward node that will never be called.  The `indices`
// storage is moved into the backward node so it is retained for the backward
// pass without an extra copy.
TensorImplPtr
attach_index_scatter_grad(const TensorImplPtr& a, TensorImplPtr out, Storage indices, int axis) {
    if (!differentiable_dtype(a->dtype()))
        return out;
    auto bwd = std::make_shared<IndexScatterBackward>();
    bwd->grad_shape_ = out->shape();
    bwd->indices_ = std::move(indices);
    bwd->axis_ = axis;
    kernel::NaryKernel<IndexScatterBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace

// Sort `a` in ascending order along `axis`.  The backend's sort_select
// (topk=false) returns the sorted values and the permutation indices as a
// pair.  The indices are saved in IndexScatterBackward so the backward can
// scatter the incoming gradient back to the original positions.
TensorImplPtr sort_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "sort.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"sort", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Shape out_shape = a->shape();
    // sort_select(topk=false) returns all elements in sorted order.
    auto [values, indices] = backend::Dispatcher::for_device(device).sort_select(
        a->storage(), a->shape(), out_shape, ax, dt, false);
    auto out = fresh(std::move(values), std::move(out_shape), dt, device);
    return attach_index_scatter_grad(a, std::move(out), std::move(indices), ax);
}

// Compute the permutation indices that would sort `a` along `axis` in
// ascending order.  The output always has dtype I32 and carries no gradient
// because index tensors are discrete and non-differentiable.
TensorImplPtr argsort_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "argsort.a").non_null();
    const Device device = a->device();
    OpScopeFull scope{"argsort", device, a->dtype(), a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Storage out =
        backend::Dispatcher::for_device(device).argsort(a->storage(), a->shape(), ax, a->dtype());
    return fresh(std::move(out), a->shape(), Dtype::I32, device);
}

namespace {
// Shared forward implementation for argmax (is_min=false) and argmin
// (is_min=true).  Returns an I64 index tensor whose shape is `a.shape` with
// dimension `axis` removed (or set to 1 when keepdims=true).
TensorImplPtr
argext_dispatch(const TensorImplPtr& a, int axis, bool keepdims, bool is_min, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Device device = a->device();
    OpScopeFull scope{name, device, a->dtype(), a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Shape out_shape = a->shape();
    if (keepdims)
        out_shape[ax] = 1;
    else
        out_shape.erase(out_shape.begin() + ax);
    Storage out = backend::Dispatcher::for_device(device).arg_reduce_index(
        a->storage(), a->shape(), ax, keepdims, a->dtype(), is_min);
    return fresh(std::move(out), std::move(out_shape), Dtype::I64, device);
}
}  // namespace

// Return the index of the maximum element along `axis`; delegates to argext_dispatch.
TensorImplPtr argmax_op(const TensorImplPtr& a, int axis, bool keepdims) {
    return argext_dispatch(a, axis, keepdims, false, "argmax");
}
// Return the index of the minimum element along `axis`; delegates to argext_dispatch.
TensorImplPtr argmin_op(const TensorImplPtr& a, int axis, bool keepdims) {
    return argext_dispatch(a, axis, keepdims, true, "argmin");
}

// Find all non-zero element positions in `a`.  The data is always transferred
// to CPU before counting so that the dynamic output size (number of non-zero
// elements) can be determined before allocating the output buffer.
// Returns a 2-D tensor of shape (count, ndim) with dtype I64, where each row
// is the multi-dimensional index of one non-zero element in row-major order.
TensorImplPtr nonzero_op(const TensorImplPtr& a) {
    Validator::input(a, "nonzero.a").non_null();
    const std::size_t ndim = a->shape().size();
    OpScopeFull scope{"nonzero", a->device(), a->dtype(), a->shape()};

    // Materialise on CPU so we can count non-zeros before allocating output.
    CpuStorage cpu = backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());

    std::size_t count = 0;
    CpuStorage out = backend::Dispatcher::for_device(Device::CPU)
                         .nonzero_forward(Storage{cpu}, a->shape(), a->dtype(), count);

    Shape out_shape{static_cast<std::int64_t>(count), static_cast<std::int64_t>(ndim)};
    return fresh(Storage{std::move(out)}, std::move(out_shape), Dtype::I64, Device::CPU);
}

// Sort and deduplicate all elements of `a`, returning them as a 1-D CPU
// tensor of the same dtype.  Always produces output on Device::CPU regardless
// of the input device, since std::sort and std::unique operate on host memory.
//
// The template lambda `run` is instantiated for each supported element type
// (F32, F64, I32, I64) via explicit casts from the raw byte buffer.
TensorImplPtr unique_op(const TensorImplPtr& a) {
    Validator::input(a, "unique.a").non_null();
    const Dtype dt = a->dtype();
    OpScopeFull scope{"unique", a->device(), dt, a->shape()};

    CpuStorage cpu = backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());
    const std::size_t n = shape_numel(a->shape());

    auto run = [&](const auto* src) -> CpuStorage {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(src)>>;
        std::vector<T> vals(src, src + n);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        CpuStorage out;
        out.dtype = dt;
        out.nbytes = vals.size() * sizeof(T);
        out.ptr = allocate_aligned_bytes(out.nbytes);
        std::memcpy(out.ptr.get(), vals.data(), out.nbytes);
        return out;
    };

    CpuStorage out_cpu;
    Shape out_shape;
    // After deduplication, back-compute the output element count from the
    // remaining byte count and record the 1-D output shape.
    auto wrap = [&](auto&& s) {
        out_shape = {static_cast<std::int64_t>(s.nbytes / dtype_size(dt))};
        out_cpu = std::move(s);
    };
    if (dt == Dtype::F32)
        wrap(run(reinterpret_cast<const float*>(cpu.ptr.get())));
    else if (dt == Dtype::F64)
        wrap(run(reinterpret_cast<const double*>(cpu.ptr.get())));
    else if (dt == Dtype::I32)
        wrap(run(reinterpret_cast<const std::int32_t*>(cpu.ptr.get())));
    else if (dt == Dtype::I64)
        wrap(run(reinterpret_cast<const std::int64_t*>(cpu.ptr.get())));
    else
        ErrorBuilder("unique").not_implemented("dtype not supported");
    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, Device::CPU);
}

// Return the k largest values and their original positions along `axis`.
// Calls sort_select(topk=true), which returns exactly the k greatest elements
// and their sort-indices.  The values output receives IndexScatterBackward for
// differentiable dtypes; the indices output (dtype I32) carries no gradient.
//
// The indices_out is constructed first so its storage can be shared by
// attach_index_scatter_grad without an extra allocation.
std::vector<TensorImplPtr> topk_op(const TensorImplPtr& a, std::int64_t k, int axis) {
    Validator::input(a, "topk.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"topk", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    if (k <= 0 || k > a->shape()[static_cast<std::size_t>(ax)])
        ErrorBuilder("topk").fail("k must be in (0, axis_size]");
    Shape out_shape = a->shape();
    out_shape[static_cast<std::size_t>(ax)] = k;
    auto [values, indices] = backend::Dispatcher::for_device(device).sort_select(
        a->storage(), a->shape(), out_shape, ax, dt, true);
    // Build indices_out before values_out so that its storage pointer remains
    // valid when it is borrowed by attach_index_scatter_grad below.
    auto indices_out = fresh(std::move(indices), out_shape, Dtype::I32, device);
    auto values_out = fresh(std::move(values), out_shape, dt, device);
    values_out = attach_index_scatter_grad(a, std::move(values_out), indices_out->storage(), ax);
    return {std::move(values_out), std::move(indices_out)};
}

LUCID_REGISTER_OP(IndexScatterBackward)

}  // namespace lucid
