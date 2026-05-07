// lucid/_C/ops/ufunc/Scan.cpp
//
// Cumulative scan forward and backward pass implementations.
//
// Two anonymous-namespace helpers wrap the backend calls to keep apply()
// implementations readable:
//   reverse_along_axis_storage — reverses elements along one axis.
//   cumsum_storage_along       — inclusive prefix-sum along one axis.
//
// Backward for cumsum (reverse-cumsum trick):
//   Given dL/dy (upstream gradient of the output), the gradient for position i
//   is the sum of dL/dy_j for all j >= i.  This equals:
//     dx = reverse(cumsum(reverse(dy)))
//   which avoids an explicit O(n^2) summation.
//
// Backward for cumprod:
//   dx_i = sum_j [ dy_j * (prod_y_j / x_i) ]
//   where prod_y_j = saved_y_[j] (the forward cumprod output).  Reorganising:
//     dx = reverse(cumsum(reverse(dy * saved_y))) / saved_x
//   This reuses the reverse-cumsum primitive applied to (dy * saved_y),
//   then divides element-wise by the saved input x.

#include "Scan.h"

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

// Thin wrapper: reverse the storage contents along the given axis.
Storage reverse_along_axis_storage(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).reverse_along_axis(s, shape, axis, dt);
}

// Thin wrapper: compute the inclusive prefix sum along the given axis.
Storage
cumsum_storage_along(const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    return backend::Dispatcher::for_device(device).cumsum(s, shape, axis, dt);
}

// Private backward node for cumsum.
//
// Saved state:
//   input_shape_ — shape of the forward input (= shape of the output).
//   axis_        — normalised (non-negative) reduction axis.
class CumsumBackward : public AutogradNode<CumsumBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    // dx = reverse(cumsum(reverse(dy))):
    // 1. Reverse dy along axis to turn suffix-sums into prefix-sums.
    // 2. Apply cumsum to accumulate the reversed gradients.
    // 3. Reverse again to restore the original axis order.
    std::vector<Storage> apply(Storage grad_out) override {
        Storage rev = reverse_along_axis_storage(grad_out, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage dx = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        return {std::move(dx)};
    }
};

// Private backward node for cumprod.
//
// Saved state:
//   input_shape_ — shape of the forward input.
//   axis_        — normalised reduction axis.
//   saved_x_     — copy of the forward input tensor's storage.
//   saved_y_     — copy of the forward output (cumprod) tensor's storage.
class CumprodBackward : public AutogradNode<CumprodBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;
    Storage saved_x_;
    Storage saved_y_;

    // dx = reverse(cumsum(reverse(dy * saved_y))) / saved_x.
    // Step 1: weight each upstream gradient by the corresponding cumprod output.
    // Steps 2-3: apply the reverse-cumsum trick to accumulate weighted gradients.
    // Step 4: divide by the saved input x to obtain the per-element gradient.
    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t total = shape_numel(input_shape_);
        // dy * y  (element-wise product of gradient and cumprod output)
        Storage p_s = multiply_storages(grad_out, saved_y_, total, dtype_, device_);
        Storage rev = reverse_along_axis_storage(p_s, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage q = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        Storage dx = divide_storages(q, saved_x_, total, dtype_, device_);
        return {std::move(dx)};
    }
};

// KeepInput: scans are valid for integer types; no promotion needed.
const OpSchema CumsumBackward::schema_v1{"cumsum", 1, AmpPolicy::KeepInput, true};
const OpSchema CumprodBackward::schema_v1{"cumprod", 1, AmpPolicy::KeepInput, true};

// Shared forward dispatch for both scan ops.  Validates the input, normalises
// the axis, dispatches to the backend, and returns a fresh output tensor.
// Autograd wiring is left to the callers so that each can save the appropriate
// tensors (cumsum saves nothing extra; cumprod saves x and y).
TensorImplPtr scan_dispatch(const TensorImplPtr& a, int axis, bool is_prod, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    auto sh = a->shape();
    if (sh.empty())
        ErrorBuilder(name).fail("input is scalar");
    int ax = axis;
    if (ax < 0)
        ax += static_cast<int>(sh.size());
    if (ax < 0 || ax >= (int)sh.size())
        ErrorBuilder(name).fail("axis out of range");
    OpScopeFull scope{name, device, dt, sh};

    Storage out_storage =
        is_prod ? backend::Dispatcher::for_device(device).cumprod(a->storage(), sh, ax, dt)
                : backend::Dispatcher::for_device(device).cumsum(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

}  // namespace

// Dispatch cumsum, then wire CumsumBackward.  The axis is re-normalised here
// (after scan_dispatch validated it) so that bwd->axis_ is always non-negative.
TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, false, "cumsum");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumsumBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    kernel::NaryKernel<CumsumBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// Dispatch cumprod, then wire CumprodBackward with both input and output saved.
TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, true, "cumprod");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumprodBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->saved_x_ = a->storage();    // needed for the final division in apply()
    bwd->saved_y_ = out->storage();  // cumprod output, used as the weight
    kernel::NaryKernel<CumprodBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

// ─── cummax / cummin backward ───────────────────────────────────────────────
//
// Backward for cummax: dy[k] flows to the position that first achieved
// the running maximum up to position k.  We process right-to-left along
// the scan axis, accumulating dy into a running sum that is "claimed" (reset)
// whenever we hit a new-max position (i.e., saved_y_[k] > saved_y_[k-1]).
//
// The is_new_max condition at k is:
//   k == 0  OR  saved_y_[k] > saved_y_[k-1]   (cummax is non-decreasing)
//
// Same logic applies for cummin with "saved_y_[k] < saved_y_[k-1]".

namespace {

// Dispatch cummax or cummin forward via the backend.
TensorImplPtr scan_ext_dispatch(const TensorImplPtr& a, int axis,
                                bool is_max, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    auto sh = a->shape();
    if (sh.empty())
        ErrorBuilder(name).fail("input is scalar");
    int ax = axis;
    if (ax < 0)
        ax += static_cast<int>(sh.size());
    if (ax < 0 || ax >= (int)sh.size())
        ErrorBuilder(name).fail("axis out of range");
    OpScopeFull scope{name, device, dt, sh};

    Storage out_storage =
        is_max ? backend::Dispatcher::for_device(device).cummax(a->storage(), sh, ax, dt)
               : backend::Dispatcher::for_device(device).cummin(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

// Generic segmented right-to-left accumulation.
// Template parameter IsMax selects the is_new_extreme condition.
template <bool IsMax, typename T>
void scan_ext_backward_loop(const T* dy, const T* y,
                            T* dx, const Shape& shape, int axis) {
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            T running_sum = T(0);
            // Right-to-left pass
            for (std::size_t k = L; k-- > 0;) {
                std::size_t idx = (o * L + k) * inner + j;
                running_sum += dy[idx];
                // is_new_extreme: first position (k==0) or y[k] changed from y[k-1]
                bool is_new = (k == 0);
                if (!is_new) {
                    std::size_t prev_idx = (o * L + k - 1) * inner + j;
                    if constexpr (IsMax)
                        is_new = (y[idx] > y[prev_idx]);
                    else
                        is_new = (y[idx] < y[prev_idx]);
                }
                if (is_new) {
                    dx[idx] = running_sum;
                    running_sum = T(0);
                } else {
                    dx[idx] = T(0);
                }
            }
        }
    }
}

Storage cummax_backward_cpu(const Storage& grad, const Storage& out,
                             const Shape& shape, int axis, Dtype dt) {
    const auto& g_cs = std::get<CpuStorage>(grad);
    const auto& o_cs = std::get<CpuStorage>(out);
    std::size_t nb = g_cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    if (dt == Dtype::F32)
        scan_ext_backward_loop<true>(
            reinterpret_cast<const float*>(g_cs.ptr.get()),
            reinterpret_cast<const float*>(o_cs.ptr.get()),
            reinterpret_cast<float*>(ptr.get()), shape, axis);
    else if (dt == Dtype::F64)
        scan_ext_backward_loop<true>(
            reinterpret_cast<const double*>(g_cs.ptr.get()),
            reinterpret_cast<const double*>(o_cs.ptr.get()),
            reinterpret_cast<double*>(ptr.get()), shape, axis);
    else
        ErrorBuilder("cummax_backward").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage cummin_backward_cpu(const Storage& grad, const Storage& out,
                             const Shape& shape, int axis, Dtype dt) {
    const auto& g_cs = std::get<CpuStorage>(grad);
    const auto& o_cs = std::get<CpuStorage>(out);
    std::size_t nb = g_cs.nbytes;
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);

    if (dt == Dtype::F32)
        scan_ext_backward_loop<false>(
            reinterpret_cast<const float*>(g_cs.ptr.get()),
            reinterpret_cast<const float*>(o_cs.ptr.get()),
            reinterpret_cast<float*>(ptr.get()), shape, axis);
    else if (dt == Dtype::F64)
        scan_ext_backward_loop<false>(
            reinterpret_cast<const double*>(g_cs.ptr.get()),
            reinterpret_cast<const double*>(o_cs.ptr.get()),
            reinterpret_cast<double*>(ptr.get()), shape, axis);
    else
        ErrorBuilder("cummin_backward").not_implemented("dtype not supported");

    return Storage{CpuStorage{ptr, nb, dt}};
}

// Backward node for cummax.
class CummaxBackward : public AutogradNode<CummaxBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;
    Storage saved_y_;  // cummax output — needed to compute is_new_max

    std::vector<Storage> apply(Storage grad_out) override {
        // For GPU devices: evaluate MLX arrays, compute loop on CPU, rewrap.
        if (device_ == Device::GPU) {
            const auto& g_gs = std::get<GpuStorage>(grad_out);
            const auto& y_gs = std::get<GpuStorage>(saved_y_);
            g_gs.arr->eval();
            y_gs.arr->eval();
            auto g_cont = ::mlx::core::contiguous(*g_gs.arr);
            auto y_cont = ::mlx::core::contiguous(*y_gs.arr);
            g_cont.eval();
            y_cont.eval();

            std::size_t n = shape_numel(input_shape_);
            auto shape_mlx = gpu::to_mlx_shape(input_shape_);
            auto mlx_dt = gpu::to_mlx_dtype(dtype_);

            if (dtype_ == Dtype::F32) {
                std::vector<float> dx(n, 0.0f);
                scan_ext_backward_loop<true>(
                    g_cont.data<float>(), y_cont.data<float>(),
                    dx.data(), input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx_arr), dtype_)}};
            } else if (dtype_ == Dtype::F64) {
                std::vector<double> dx(n, 0.0);
                scan_ext_backward_loop<true>(
                    g_cont.data<double>(), y_cont.data<double>(),
                    dx.data(), input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx_arr), dtype_)}};
            } else {
                ErrorBuilder("cummax_backward").not_implemented("dtype");
                return {};
            }
        }
        // CPU path
        return {cummax_backward_cpu(grad_out, saved_y_, input_shape_, axis_, dtype_)};
    }
};

// Backward node for cummin.
class CumminBackward : public AutogradNode<CumminBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;
    Storage saved_y_;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& g_gs = std::get<GpuStorage>(grad_out);
            const auto& y_gs = std::get<GpuStorage>(saved_y_);
            g_gs.arr->eval();
            y_gs.arr->eval();
            auto g_cont = ::mlx::core::contiguous(*g_gs.arr);
            auto y_cont = ::mlx::core::contiguous(*y_gs.arr);
            g_cont.eval();
            y_cont.eval();

            std::size_t n = shape_numel(input_shape_);
            auto shape_mlx = gpu::to_mlx_shape(input_shape_);
            auto mlx_dt = gpu::to_mlx_dtype(dtype_);

            if (dtype_ == Dtype::F32) {
                std::vector<float> dx(n, 0.0f);
                scan_ext_backward_loop<false>(
                    g_cont.data<float>(), y_cont.data<float>(),
                    dx.data(), input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx_arr), dtype_)}};
            } else if (dtype_ == Dtype::F64) {
                std::vector<double> dx(n, 0.0);
                scan_ext_backward_loop<false>(
                    g_cont.data<double>(), y_cont.data<double>(),
                    dx.data(), input_shape_, axis_);
                ::mlx::core::array dx_arr(dx.data(), shape_mlx, mlx_dt);
                return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx_arr), dtype_)}};
            } else {
                ErrorBuilder("cummin_backward").not_implemented("dtype");
                return {};
            }
        }
        return {cummin_backward_cpu(grad_out, saved_y_, input_shape_, axis_, dtype_)};
    }
};

const OpSchema CummaxBackward::schema_v1{"cummax", 1, AmpPolicy::KeepInput, true};
const OpSchema CumminBackward::schema_v1{"cummin", 1, AmpPolicy::KeepInput, true};

}  // anonymous namespace

TensorImplPtr cummax_op(const TensorImplPtr& a, int axis) {
    auto out = scan_ext_dispatch(a, axis, true, "cummax");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CummaxBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->saved_y_ = out->storage();
    kernel::NaryKernel<CummaxBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

TensorImplPtr cummin_op(const TensorImplPtr& a, int axis) {
    auto out = scan_ext_dispatch(a, axis, false, "cummin");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumminBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->saved_y_ = out->storage();
    kernel::NaryKernel<CumminBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

}  // namespace lucid
