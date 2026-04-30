#include "Scan.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

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

using ufunc_detail::allocate_cpu;
using ufunc_detail::fresh;

// Reverse a Storage along `axis` (used by cumsum backward).
Storage reverse_along_axis_storage(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    if (device == Device::GPU) {
        const auto& g = std::get<GpuStorage>(s);
        std::vector<std::int32_t> idx(shape[axis]);
        for (std::int64_t i = 0; i < shape[axis]; ++i)
            idx[i] = static_cast<std::int32_t>(shape[axis] - 1 - i);
        ::mlx::core::Shape idx_shape(shape.size(), 1);
        idx_shape[axis] = shape[axis];
        ::mlx::core::array idx_arr(idx.data(), idx_shape, ::mlx::core::int32);
        idx_arr = ::mlx::core::broadcast_to(idx_arr, gpu::to_mlx_shape(shape));
        auto out = ::mlx::core::take_along_axis(*g.arr, idx_arr, axis);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& cs = std::get<CpuStorage>(s);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = cs.nbytes;
    out.ptr = allocate_aligned_bytes(out.nbytes);
    const std::size_t elem = dtype_size(dt);
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[d]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    const std::size_t L = static_cast<std::size_t>(shape[axis]);
    for (std::size_t o = 0; o < outer; ++o)
        for (std::size_t k = 0; k < L; ++k)
            std::memcpy(out.ptr.get() + ((o * L + k) * inner) * elem,
                        cs.ptr.get() + ((o * L + (L - 1 - k)) * inner) * elem, inner * elem);
    return Storage{std::move(out)};
}

Storage cumsum_storage_along(
    const Storage& s, const Shape& shape, int axis, Dtype dt, Device device) {
    if (device == Device::GPU) {
        const auto& g = std::get<GpuStorage>(s);
        auto out = ::mlx::core::cumsum(*g.arr, axis);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& cs = std::get<CpuStorage>(s);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = cs.nbytes;
    out.ptr = allocate_aligned_bytes(out.nbytes);
    const int ndim = static_cast<int>(shape.size());
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[d]);
    for (int d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    const std::size_t L = static_cast<std::size_t>(shape[axis]);
    auto run = [&](auto* dst, const auto* src) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                T acc = src[(o * L) * inner + j];
                dst[(o * L) * inner + j] = acc;
                for (std::size_t k = 1; k < L; ++k) {
                    acc = acc + src[(o * L + k) * inner + j];
                    dst[(o * L + k) * inner + j] = acc;
                }
            }
    };
    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(out.ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(out.ptr.get()),
            reinterpret_cast<const double*>(cs.ptr.get()));
    else
        ErrorBuilder("cumsum").not_implemented("dtype not supported");
    return Storage{std::move(out)};
}

class CumsumBackward : public AutogradNode<CumsumBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;

    std::vector<Storage> apply(Storage grad_out) override {
        Storage rev = reverse_along_axis_storage(grad_out, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage dx = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        return {std::move(dx)};
    }
};

// cumprod backward.
//   y_k = x_0 * x_1 * ... * x_k
//   dx_j = sum_{k >= j} (g_k * y_k) / x_j
//        = reverse_cumsum(g * y)_j / x_j
// We materialize this via:
//   1. forward y = cumprod(x, axis)            (saved at forward time)
//   2. p = g * y
//   3. q = reverse_cumsum(p, axis)
//   4. dx = q / x   (NaN at zeros — same convention as PyTorch)
class CumprodBackward : public AutogradNode<CumprodBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    int axis_;
    Storage saved_x_;
    Storage saved_y_;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& gx = std::get<GpuStorage>(saved_x_);
            const auto& gy = std::get<GpuStorage>(saved_y_);
            const auto& gg = std::get<GpuStorage>(grad_out);
            auto p = ::mlx::core::multiply(*gg.arr, *gy.arr);
            // reverse along axis, cumsum, reverse back
            std::vector<std::int32_t> idx(input_shape_[axis_]);
            for (std::int64_t i = 0; i < input_shape_[axis_]; ++i)
                idx[i] = static_cast<std::int32_t>(input_shape_[axis_] - 1 - i);
            ::mlx::core::Shape idx_shape(input_shape_.size(), 1);
            idx_shape[axis_] = input_shape_[axis_];
            ::mlx::core::array idx_arr(idx.data(), idx_shape, ::mlx::core::int32);
            idx_arr = ::mlx::core::broadcast_to(idx_arr, gpu::to_mlx_shape(input_shape_));
            auto p_rev = ::mlx::core::take_along_axis(p, idx_arr, axis_);
            auto cs = ::mlx::core::cumsum(p_rev, axis_);
            auto q = ::mlx::core::take_along_axis(cs, idx_arr, axis_);
            auto dx = ::mlx::core::divide(q, *gx.arr);
            return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)}};
        }
        // CPU path: same recipe via the existing reverse/cumsum helpers.
        const auto& cx = std::get<CpuStorage>(saved_x_);
        const auto& cy = std::get<CpuStorage>(saved_y_);
        const auto& cg = std::get<CpuStorage>(grad_out);

        // p = g * y (allocate fresh)
        CpuStorage p;
        p.dtype = dtype_;
        p.nbytes = cg.nbytes;
        p.ptr = allocate_aligned_bytes(p.nbytes);
        const std::size_t total = shape_numel(input_shape_);
        auto mul_kernel = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* gp = reinterpret_cast<const T*>(cg.ptr.get());
            const T* yp = reinterpret_cast<const T*>(cy.ptr.get());
            T* op = reinterpret_cast<T*>(p.ptr.get());
            for (std::size_t i = 0; i < total; ++i)
                op[i] = gp[i] * yp[i];
        };
        if (dtype_ == Dtype::F32)
            mul_kernel(float{});
        else if (dtype_ == Dtype::F64)
            mul_kernel(double{});
        else
            ErrorBuilder("cumprod backward").not_implemented("dtype not supported");

        Storage p_s{std::move(p)};
        Storage rev = reverse_along_axis_storage(p_s, input_shape_, axis_, dtype_, device_);
        Storage cs = cumsum_storage_along(rev, input_shape_, axis_, dtype_, device_);
        Storage q = reverse_along_axis_storage(cs, input_shape_, axis_, dtype_, device_);
        // dx = q / x
        const auto& cq = std::get<CpuStorage>(q);
        CpuStorage dx;
        dx.dtype = dtype_;
        dx.nbytes = cq.nbytes;
        dx.ptr = allocate_aligned_bytes(dx.nbytes);
        auto div_kernel = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* qp = reinterpret_cast<const T*>(cq.ptr.get());
            const T* xp = reinterpret_cast<const T*>(cx.ptr.get());
            T* dp = reinterpret_cast<T*>(dx.ptr.get());
            for (std::size_t i = 0; i < total; ++i)
                dp[i] = qp[i] / xp[i];
        };
        if (dtype_ == Dtype::F32)
            div_kernel(float{});
        else
            div_kernel(double{});
        return {Storage{std::move(dx)}};
    }
};

const OpSchema CumsumBackward::schema_v1{"cumsum", 1, AmpPolicy::KeepInput, true};
const OpSchema CumprodBackward::schema_v1{"cumprod", 1, AmpPolicy::KeepInput, true};

template <typename T, bool IsProd>
void scan_axis(const T* in, T* out, const Shape& shape, int axis) {
    const std::size_t ndim = shape.size();
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[d]);
    for (std::size_t d = axis + 1; d < ndim; ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    const std::size_t L = static_cast<std::size_t>(shape[axis]);

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            const std::size_t base = o * L * inner + j;
            T acc = in[base];
            out[base] = acc;
            for (std::size_t k = 1; k < L; ++k) {
                const std::size_t idx = base + k * inner;
                if constexpr (IsProd)
                    acc = acc * in[idx];
                else
                    acc = acc + in[idx];
                out[idx] = acc;
            }
        }
    }
}

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

    Storage out_storage = is_prod
                              ? backend::Dispatcher::for_device(device).cumprod(a->storage(), sh, ax, dt)
                              : backend::Dispatcher::for_device(device).cumsum(a->storage(), sh, ax, dt);
    return fresh(std::move(out_storage), sh, dt, device);
}

}  // namespace

TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, /*is_prod=*/false, "cumsum");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumsumBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    kernel::NaryKernel<CumsumBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                         /*save_ins=*/false);
    return out;
}

TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis) {
    auto out = scan_dispatch(a, axis, /*is_prod=*/true, "cumprod");
    int ax = axis < 0 ? axis + (int)a->shape().size() : axis;
    auto bwd = std::make_shared<CumprodBackward>();
    bwd->input_shape_ = a->shape();
    bwd->axis_ = ax;
    bwd->saved_x_ = a->storage();
    bwd->saved_y_ = out->storage();
    kernel::NaryKernel<CumprodBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                          /*save_ins=*/false);
    return out;
}

}  // namespace lucid
