#include "Pad.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
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
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::numel;

CpuStorage crop_pad_cpu(const CpuStorage& grad,
                        const Shape& input_shape,
                        const Shape& output_shape,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        Dtype dt) {
    auto out = allocate_cpu(input_shape, dt);
    if (input_shape.empty()) {
        if (out.nbytes > 0) {
            std::memcpy(out.ptr.get(), grad.ptr.get(), out.nbytes);
        }
        return out;
    }

    const std::size_t ndim = input_shape.size();
    const std::size_t elem = dtype_size(dt);
    Stride in_stride(ndim), out_stride(ndim);
    in_stride.back() = 1;
    out_stride.back() = 1;
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
        in_stride[static_cast<std::size_t>(d)] = in_stride[static_cast<std::size_t>(d) + 1] *
                                                 input_shape[static_cast<std::size_t>(d) + 1];
        out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                  output_shape[static_cast<std::size_t>(d) + 1];
    }

    const std::size_t row_in = static_cast<std::size_t>(input_shape.back());
    const std::size_t row_bytes = row_in * elem;
    const std::size_t rows = numel(input_shape) / row_in;
    std::vector<std::int64_t> coord(ndim - 1, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t src_off = static_cast<std::size_t>(pad_width.back().first);
        for (std::size_t d = 0; d + 1 < ndim; ++d) {
            src_off += static_cast<std::size_t>(coord[d] + pad_width[d].first) *
                       static_cast<std::size_t>(out_stride[d]);
        }
        std::size_t dst_off = 0;
        for (std::size_t d = 0; d + 1 < ndim; ++d) {
            dst_off += static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
        }
        std::memcpy(out.ptr.get() + dst_off * elem, grad.ptr.get() + src_off * elem, row_bytes);
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < input_shape[static_cast<std::size_t>(d)]) {
                break;
            }
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return out;
}

class PadBackward : public FuncOp<PadBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<std::pair<std::int64_t, std::int64_t>> pad_width_;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& g = std::get<GpuStorage>(grad_out);
            ::mlx::core::Shape lo(input_shapes_[0].size(), 0);
            ::mlx::core::Shape hi;
            hi.reserve(input_shapes_[0].size());
            for (std::size_t d = 0; d < input_shapes_[0].size(); ++d) {
                lo[d] = static_cast<::mlx::core::ShapeElem>(pad_width_[d].first);
                hi.push_back(
                    static_cast<::mlx::core::ShapeElem>(pad_width_[d].first + input_shapes_[0][d]));
            }
            auto out = ::mlx::core::slice(*g.arr, lo, hi);
            out = ::mlx::core::contiguous(out);
            return {Storage{gpu::wrap_mlx_array(std::move(out), dtype_)}};
        }
        return {Storage{crop_pad_cpu(std::get<CpuStorage>(grad_out), input_shapes_[0], out_shape_,
                                     pad_width_, dtype_)}};
    }
};

const OpSchema PadBackward::schema_v1{"pad", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_pad_grad(const TensorImplPtr& a,
                              TensorImplPtr out,
                              std::vector<std::pair<std::int64_t, std::int64_t>> pad_width) {
    auto bwd = std::make_shared<PadBackward>();
    bwd->pad_width_ = std::move(pad_width);
    kernel::NaryKernel<PadBackward, 1>::wire_autograd(std::move(bwd), {a}, out, /*save_ins=*/false);
    return out;
}

LUCID_REGISTER_OP(PadBackward)

}  // namespace

TensorImplPtr pad_op(const TensorImplPtr& a,
                     std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
                     double constant) {
    Validator::input(a, "pad.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"pad", device, dt, a->shape()};
    if (pad_width.size() != a->shape().size())
        ErrorBuilder("pad").fail("pad_width length must equal ndim");
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        std::vector<std::pair<int, int>> mlx_pad;
        mlx_pad.reserve(pad_width.size());
        for (auto& [lo, hi] : pad_width)
            mlx_pad.emplace_back(static_cast<int>(lo), static_cast<int>(hi));
        ::mlx::core::array pad_value(static_cast<float>(constant), gpu::to_mlx_dtype(dt));
        auto out = ::mlx::core::pad(*ga.arr, mlx_pad, pad_value);
        Shape sh = mlx_shape_to_lucid(out.shape());
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(sh), dt, device);
        return attach_pad_grad(a, std::move(result), std::move(pad_width));
    }
    const std::size_t ndim = a->shape().size();
    Shape out_shape(ndim);
    for (std::size_t d = 0; d < ndim; ++d)
        out_shape[d] = a->shape()[d] + pad_width[d].first + pad_width[d].second;

    auto out_cpu = allocate_cpu(out_shape, dt);
    auto fill = [&](auto* dst, std::size_t n, double value) {
        using T = std::remove_pointer_t<decltype(dst)>;
        const T v = static_cast<T>(value);
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = v;
    };
    const std::size_t out_numel = numel(out_shape);
    if (constant != 0.0) {
        switch (dt) {
            case Dtype::F32:
                fill(reinterpret_cast<float*>(out_cpu.ptr.get()), out_numel, constant);
                break;
            case Dtype::F64:
                fill(reinterpret_cast<double*>(out_cpu.ptr.get()), out_numel, constant);
                break;
            case Dtype::I32:
                fill(reinterpret_cast<std::int32_t*>(out_cpu.ptr.get()), out_numel, constant);
                break;
            case Dtype::I64:
                fill(reinterpret_cast<std::int64_t*>(out_cpu.ptr.get()), out_numel, constant);
                break;
            default:
                ErrorBuilder("pad").not_implemented("dtype not supported");
        }
    }

    const auto& ca = std::get<CpuStorage>(a->storage());
    const std::size_t elem = dtype_size(dt);
    Stride in_stride(ndim), out_stride(ndim);
    if (ndim > 0) {
        in_stride.back() = 1;
        out_stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 2; d >= 0; --d) {
            in_stride[d] = in_stride[d + 1] * a->shape()[d + 1];
            out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
        }
    }
    const std::size_t row_in = static_cast<std::size_t>(a->shape().back());
    const std::size_t row_bytes = row_in * elem;
    const std::size_t in_numel = numel(a->shape());
    const std::size_t rows = in_numel / row_in;

    std::vector<std::int64_t> coord(ndim - 1, 0);
    for (std::size_t r = 0; r < rows; ++r) {
        std::size_t out_off = pad_width.back().first;
        for (std::size_t d = 0; d + 1 < ndim; ++d)
            out_off += static_cast<std::size_t>(coord[d] + pad_width[d].first) *
                       static_cast<std::size_t>(out_stride[d]);
        std::size_t in_off = 0;
        for (std::size_t d = 0; d + 1 < ndim; ++d)
            in_off += static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
        std::memcpy(out_cpu.ptr.get() + out_off * elem, ca.ptr.get() + in_off * elem, row_bytes);
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 2; d >= 0; --d) {
            if (++coord[d] < a->shape()[d])
                break;
            coord[d] = 0;
        }
    }

    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    return attach_pad_grad(a, std::move(result), std::move(pad_width));
}

}  // namespace lucid
