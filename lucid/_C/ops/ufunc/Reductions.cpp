// lucid/_C/ops/ufunc/Reductions.cpp
//
// CPU and GPU reduction kernels and gradient formulas for sum, mean, prod,
// max, min.  The file also contains two anonymous-namespace helpers shared by
// all CPU paths:
//   oir_for_axis       — decomposes a shape into (outer, reduce_dim, inner)
//                        factors for a single axis, used by the strided loop.
//   reduce_one_axis    — applies an Accelerate-backed 1-D reduction kernel over
//                        the inner/outer loop structure for one axis at a time.
//   multi_axis_reduce  — chains reduce_one_axis over a sorted axis list in
//                        descending order so indices remain valid after each
//                        dimension is removed.

#include "Reductions.h"

#include "Exponential.h"
#include "Var.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../../backend/cpu/Reduce.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/OpRegistry.h"

namespace lucid {

namespace {

// Outer / reduce_dim / inner decomposition of a shape for a given axis.
// outer   = product of all dims before axis
// reduce  = shape[axis]
// inner   = product of all dims after axis
struct OIR {
    std::size_t outer;
    std::size_t reduce_dim;
    std::size_t inner;
};

// Compute OIR factors so the strided reduction kernel knows how to traverse
// the input buffer without building an explicit index array.
OIR oir_for_axis(const Shape& shape, int axis) {
    OIR r{1, static_cast<std::size_t>(shape[axis]), 1};
    for (int d = 0; d < axis; ++d)
        r.outer *= static_cast<std::size_t>(shape[d]);
    for (std::size_t d = axis + 1; d < shape.size(); ++d)
        r.inner *= static_cast<std::size_t>(shape[d]);
    return r;
}

// Intermediate result of a single-axis reduction (data + new shape).
struct AxisResult {
    CpuStorage data;
    Shape shape;
};

// Apply a typed Accelerate kernel (k32 for F32, k64 for F64) along one axis.
// Allocates the output buffer, then calls the appropriate typed overload.
// Only F32 and F64 are supported; other dtypes raise a not-implemented error.
template <class Kernel32, class Kernel64>
AxisResult reduce_one_axis(const CpuStorage& in,
                           const Shape& in_shape,
                           int axis,
                           Dtype dt,
                           Kernel32 k32,
                           Kernel64 k64,
                           const char* op_name) {
    const auto oir = oir_for_axis(in_shape, axis);
    Shape out_shape = in_shape;
    out_shape.erase(out_shape.begin() + axis);
    const std::size_t out_numel = (oir.outer * oir.inner);

    AxisResult r;
    r.shape = std::move(out_shape);
    r.data.dtype = dt;
    r.data.nbytes = out_numel * dtype_size(dt);
    r.data.ptr = allocate_aligned_bytes(r.data.nbytes);

    switch (dt) {
    case Dtype::F32:
        k32(reinterpret_cast<const float*>(in.ptr.get()),
            reinterpret_cast<float*>(r.data.ptr.get()), oir.outer, oir.reduce_dim, oir.inner);
        break;
    case Dtype::F64:
        k64(reinterpret_cast<const double*>(in.ptr.get()),
            reinterpret_cast<double*>(r.data.ptr.get()), oir.outer, oir.reduce_dim, oir.inner);
        break;
    default:
        ErrorBuilder(op_name).not_implemented("dtype not supported (F32/F64 only)");
    }
    return r;
}

// Reduce over multiple axes by iterating them in descending order.
// Descending order ensures that lower axis indices remain valid as dimensions
// are successively removed from current_shape after each single-axis pass.
// keepdims support is tracked but the actual shape expansion happens at the
// ReduceKernel level; here we only care about the final data layout.
template <class Kernel32, class Kernel64>
CpuStorage multi_axis_reduce(const CpuStorage& a,
                             const Shape& in_shape,
                             const std::vector<int>& axes,
                             bool keepdims,
                             Dtype dt,
                             Kernel32 k32,
                             Kernel64 k64,
                             const char* op_name) {
    std::vector<int> ax_desc(axes.rbegin(), axes.rend());

    CpuStorage current = a;
    Shape current_shape = in_shape;
    for (int ax : ax_desc) {
        auto r = reduce_one_axis(current, current_shape, ax, dt, k32, k64, op_name);
        current = std::move(r.data);
        current_shape = std::move(r.shape);
    }

    if (keepdims) {
        // keepdims shape construction is handled by the ReduceKernel caller;
        // the kept variable below would hold it but is intentionally unused.
        Shape kept = in_shape;
        for (int ax : axes)
            kept[ax] = 1;

        (void)kept;
    }
    return current;
}

}  // namespace

// sum — broadcast-back is O(n) and requires no saved tensors.
const OpSchema SumBackward::schema_v1{"sum", 1, AmpPolicy::Promote, true};

// dL/dx = broadcast(dL/dy) back to the original input shape along reduce_axes_.
Storage SumBackward::grad_formula(const Storage& grad_out) {
    return broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                     this->reduce_axes_, this->keepdims_, this->dtype_,
                                     this->device_);
}

TensorImplPtr sum_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return SumBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(SumBackward)

// mean — uses AmpPolicy::Promote; divides broadcast gradient by the count of
// reduced elements.
const OpSchema MeanBackward::schema_v1{"mean", 1, AmpPolicy::Promote, true};

namespace {
// Count the number of elements collapsed by the given axes.
double reduced_count(const Shape& in_shape, const std::vector<int>& axes) {
    double n = 1.0;
    for (int ax : axes)
        n *= static_cast<double>(in_shape[ax]);
    return n;
}
}  // namespace

// dL/dx = broadcast(dL/dy) / N  where N is the number of reduced elements.
Storage MeanBackward::grad_formula(const Storage& grad_out) {
    Storage broadcasted =
        broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    const double n = reduced_count(this->full_input_shape_, this->reduce_axes_);
    return mul_scalar_storage(broadcasted, 1.0 / n, shape_numel(this->full_input_shape_),
                              this->dtype_, this->device_);
}

TensorImplPtr mean_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return MeanBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(MeanBackward)

// prod — CPU uses Accelerate via multi_axis_reduce; GPU uses mlx::core::prod.
const OpSchema ProdBackward::schema_v1{"prod", 1, AmpPolicy::Promote, true};

// Apply the multi_axis_reduce helper with Accelerate prod kernels.
CpuStorage ProdBackward::cpu_kernel(const CpuStorage& a,
                                    const Shape& in_shape,
                                    const std::vector<int>& axes,
                                    bool keepdims,
                                    Dtype dt) {
    return multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::prod_axis_f32,
                             backend::cpu::prod_axis_f64, "prod");
}

// dL/dx_i = dL/dy * (prod_y / x_i).
// Both broadcast_back calls expand the reduced dimensions to match the input.
// ratio = out_bcast / input represents the "product of all others" for each x_i.
Storage ProdBackward::grad_formula(const Storage& grad_out) {
    const std::size_t in_numel = shape_numel(this->full_input_shape_);
    Storage g_bcast =
        broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    Storage out_bcast =
        broadcast_back_for_reduce(this->saved_output_, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    Storage ratio =
        divide_storages(out_bcast, this->saved_inputs_[0], in_numel, this->dtype_, this->device_);
    return multiply_storages(g_bcast, ratio, in_numel, this->dtype_, this->device_);
}

TensorImplPtr prod_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return ProdBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(ProdBackward)

// max — saves the output (max values) to build the equality mask cheaply.
const OpSchema MaxBackward::schema_v1{"max", 1, AmpPolicy::KeepInput, true};

// dL/dx = dL/dy * (x == max_val), broadcast back to the input shape.
// The equality mask is constructed as ge(x, m) AND ge(m, x) to avoid a
// dedicated equal-mask primitive; this handles ties correctly.
Storage MaxBackward::grad_formula(const Storage& grad_out) {
    const std::size_t in_numel = shape_numel(this->full_input_shape_);
    Storage out_bcast =
        broadcast_back_for_reduce(this->saved_output_, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);

    Storage mask_eq;
    {
        // a == b iff a >= b AND b >= a, avoiding a separate equality kernel.
        Storage ge_a = ge_mask_storage(this->saved_inputs_[0], out_bcast, in_numel, this->dtype_,
                                       this->device_);
        Storage ge_b = ge_mask_storage(out_bcast, this->saved_inputs_[0], in_numel, this->dtype_,
                                       this->device_);
        mask_eq = multiply_storages(ge_a, ge_b, in_numel, this->dtype_, this->device_);
    }

    Storage g_bcast =
        broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    return multiply_storages(g_bcast, mask_eq, in_numel, this->dtype_, this->device_);
}

TensorImplPtr max_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return MaxBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(MaxBackward)

// min — symmetric to max; the same equality-mask trick applies.
const OpSchema MinBackward::schema_v1{"min", 1, AmpPolicy::KeepInput, true};

// dL/dx = dL/dy * (x == min_val), broadcast back to the input shape.
Storage MinBackward::grad_formula(const Storage& grad_out) {
    const std::size_t in_numel = shape_numel(this->full_input_shape_);
    Storage out_bcast =
        broadcast_back_for_reduce(this->saved_output_, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    Storage mask_eq;
    {
        Storage ge_a = ge_mask_storage(this->saved_inputs_[0], out_bcast, in_numel, this->dtype_,
                                       this->device_);
        Storage ge_b = ge_mask_storage(out_bcast, this->saved_inputs_[0], in_numel, this->dtype_,
                                       this->device_);
        mask_eq = multiply_storages(ge_a, ge_b, in_numel, this->dtype_, this->device_);
    }
    Storage g_bcast =
        broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);
    return multiply_storages(g_bcast, mask_eq, in_numel, this->dtype_, this->device_);
}

TensorImplPtr min_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return MinBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(MinBackward)

namespace {
// Thin helper: validate the GPU input and delegate to an MLX lambda, then wrap
// the resulting mlx::array back into a GpuStorage.
template <class F>
GpuStorage gpu_reduce_apply(const GpuStorage& a, Dtype dt, F&& f, const char* op) {
    if (!a.arr) {
        ErrorBuilder(op).fail("null GPU input");
    }
    auto out = f(*a.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}
}  // namespace

// GPU prod: delegate directly to mlx::core::prod with the requested axes.
GpuStorage ProdBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt, [&axes, keepdims](const auto& x) { return ::mlx::core::prod(x, axes, keepdims); },
        "prod");
}

// std_op composes sqrt and var entry points so that the standard autograd graph
// handles the gradient automatically via the chain rule:
//   std = sqrt(var(x, axes, keepdims))
// No new backward node is required; SqrtBackward and VarBackward cover it.
TensorImplPtr std_op(const TensorImplPtr& a,
                     const std::vector<int>& axes,
                     bool keepdims) {
    return sqrt_op(var_op(a, axes, keepdims));
}

}  // namespace lucid
