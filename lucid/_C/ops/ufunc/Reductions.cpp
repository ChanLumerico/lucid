#include "Reductions.h"

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

// Compute (outer, reduce_dim, inner) for reducing `axis` of `shape`.
struct OIR {
    std::size_t outer;
    std::size_t reduce_dim;
    std::size_t inner;
};

OIR oir_for_axis(const Shape& shape, int axis) {
    OIR r{1, static_cast<std::size_t>(shape[axis]), 1};
    for (int d = 0; d < axis; ++d)
        r.outer *= static_cast<std::size_t>(shape[d]);
    for (std::size_t d = axis + 1; d < shape.size(); ++d)
        r.inner *= static_cast<std::size_t>(shape[d]);
    return r;
}

// Apply a 1-axis reduce kernel to (in, in_shape) at `axis`, producing a
// keepdims=False output. Returns the new buffer and the new shape.
struct AxisResult {
    CpuStorage data;
    Shape shape;
};

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

// Multi-axis reduction by sequential single-axis reduces in descending order.
// Then inserts size-1 dims if `keepdims=true`.
template <class Kernel32, class Kernel64>
CpuStorage multi_axis_reduce(const CpuStorage& a,
                             const Shape& in_shape,
                             const std::vector<int>& axes,
                             bool keepdims,
                             Dtype dt,
                             Kernel32 k32,
                             Kernel64 k64,
                             const char* op_name) {
    // axes already sorted ascending. Process descending so earlier axis
    // indices stay valid as we shrink the shape.
    std::vector<int> ax_desc(axes.rbegin(), axes.rend());

    CpuStorage current = a;  // shallow clone (shared_ptr)
    Shape current_shape = in_shape;
    for (int ax : ax_desc) {
        auto r = reduce_one_axis(current, current_shape, ax, dt, k32, k64, op_name);
        current = std::move(r.data);
        current_shape = std::move(r.shape);
    }

    if (keepdims) {
        Shape kept = in_shape;
        for (int ax : axes)
            kept[ax] = 1;
        // Data layout is identical to `current`; just swap the shape.
        // (Caller stamps it on the result TensorImpl from out_shape derived
        // from reduce_output_shape, so no further reshape needed here.)
        (void)kept;
    }
    return current;
}

}  // namespace

// =================== Sum ===================
const OpSchema SumBackward::schema_v1{"sum", 1, AmpPolicy::Promote, true};

CpuStorage SumBackward::cpu_kernel(const CpuStorage& a,
                                   const Shape& in_shape,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   Dtype dt) {
    return multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::sum_axis_f32,
                             backend::cpu::sum_axis_f64, "sum");
}

Storage SumBackward::grad_formula(const Storage& grad_out) {
    return broadcast_back_for_reduce(grad_out, this->out_shape_, this->full_input_shape_,
                                     this->reduce_axes_, this->keepdims_, this->dtype_,
                                     this->device_);
}

TensorImplPtr sum_op(const TensorImplPtr& a, const std::vector<int>& axes, bool keepdims) {
    return SumBackward::forward(a, axes, keepdims);
}
LUCID_REGISTER_OP(SumBackward)

// =================== Mean ===================
const OpSchema MeanBackward::schema_v1{"mean", 1, AmpPolicy::Promote, true};

namespace {
double reduced_count(const Shape& in_shape, const std::vector<int>& axes) {
    double n = 1.0;
    for (int ax : axes)
        n *= static_cast<double>(in_shape[ax]);
    return n;
}
}  // namespace

CpuStorage MeanBackward::cpu_kernel(const CpuStorage& a,
                                    const Shape& in_shape,
                                    const std::vector<int>& axes,
                                    bool keepdims,
                                    Dtype dt) {
    auto out = multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::sum_axis_f32,
                                 backend::cpu::sum_axis_f64, "mean");
    const double n = reduced_count(in_shape, axes);
    if (n == 0.0)
        return out;
    const double inv = 1.0 / n;
    const std::size_t numel = out.nbytes / dtype_size(dt);
    switch (dt) {
        case Dtype::F32: {
            auto* p = reinterpret_cast<float*>(out.ptr.get());
            const auto inv_f = static_cast<float>(inv);
            for (std::size_t i = 0; i < numel; ++i)
                p[i] *= inv_f;
            break;
        }
        case Dtype::F64: {
            auto* p = reinterpret_cast<double*>(out.ptr.get());
            for (std::size_t i = 0; i < numel; ++i)
                p[i] *= inv;
            break;
        }
        default:
            ErrorBuilder("mean").not_implemented("dtype not supported");
    }
    return out;
}

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

// =================== Prod ===================
const OpSchema ProdBackward::schema_v1{"prod", 1, AmpPolicy::Promote, true};

CpuStorage ProdBackward::cpu_kernel(const CpuStorage& a,
                                    const Shape& in_shape,
                                    const std::vector<int>& axes,
                                    bool keepdims,
                                    Dtype dt) {
    return multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::prod_axis_f32,
                             backend::cpu::prod_axis_f64, "prod");
}

Storage ProdBackward::grad_formula(const Storage& grad_out) {
    // dx = grad_out * (output / input) — broadcast first, then multiply.
    // Note: undefined where input has zeros; matches PyTorch's basic prod.
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

// =================== Max ===================
const OpSchema MaxBackward::schema_v1{"max", 1, AmpPolicy::KeepInput, true};

CpuStorage MaxBackward::cpu_kernel(const CpuStorage& a,
                                   const Shape& in_shape,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   Dtype dt) {
    return multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::max_axis_f32,
                             backend::cpu::max_axis_f64, "max");
}

Storage MaxBackward::grad_formula(const Storage& grad_out) {
    // 1. Broadcast saved output back to input shape (so each input position
    //    sees the max value of its reduce-block).
    const std::size_t in_numel = shape_numel(this->full_input_shape_);
    Storage out_bcast =
        broadcast_back_for_reduce(this->saved_output_, this->out_shape_, this->full_input_shape_,
                                  this->reduce_axes_, this->keepdims_, this->dtype_, this->device_);

    // 2. Mask = (input == out_bcast). Ties: every tied element receives g
    //    (matches np.maximum.gradient distributing equally across ties; can
    //    over-count gradient when ties exist — same trade-off as PyTorch's
    //    autograd reduce_max gradient at ties).
    Storage mask_eq;
    {
        // Use ge & le → both true → equal. Fast path: equal mask via two
        // comparisons. We don't have eq_mask helper; emulate with ge && ge_swap.
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

// =================== Min ===================
const OpSchema MinBackward::schema_v1{"min", 1, AmpPolicy::KeepInput, true};

CpuStorage MinBackward::cpu_kernel(const CpuStorage& a,
                                   const Shape& in_shape,
                                   const std::vector<int>& axes,
                                   bool keepdims,
                                   Dtype dt) {
    return multi_axis_reduce(a, in_shape, axes, keepdims, dt, backend::cpu::min_axis_f32,
                             backend::cpu::min_axis_f64, "min");
}

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

// =====================================================================
// GPU kernels for the reduce family (Phase 3.7.2).
// =====================================================================
//
// MLX's reduce ops accept `(arr, axes, keepdims)` and return a single
// array. We pass the axis vector through as-is (MLX accepts ascending
// order, which matches what `normalize_axes` produces).

namespace {
template <class F>
GpuStorage gpu_reduce_apply(const GpuStorage& a, Dtype dt, F&& f, const char* op) {
    if (!a.arr) {
        ErrorBuilder(op).fail("null GPU input");
    }
    auto out = f(*a.arr);
    return gpu::wrap_mlx_array(std::move(out), dt);
}
}  // namespace

GpuStorage SumBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt, [&axes, keepdims](const auto& x) { return ::mlx::core::sum(x, axes, keepdims); },
        "sum");
}

GpuStorage MeanBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt, [&axes, keepdims](const auto& x) { return ::mlx::core::mean(x, axes, keepdims); },
        "mean");
}

GpuStorage ProdBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt, [&axes, keepdims](const auto& x) { return ::mlx::core::prod(x, axes, keepdims); },
        "prod");
}

GpuStorage MaxBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt, [&axes, keepdims](const auto& x) { return ::mlx::core::max(x, axes, keepdims); },
        "max");
}

GpuStorage MinBackward::gpu_kernel(
    const GpuStorage& a, const Shape&, const std::vector<int>& axes, bool keepdims, Dtype dt) {
    return gpu_reduce_apply(
        a, dt, [&axes, keepdims](const auto& x) { return ::mlx::core::min(x, axes, keepdims); },
        "min");
}

}  // namespace lucid
