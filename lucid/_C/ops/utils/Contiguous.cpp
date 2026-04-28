#include "Contiguous.h"

#include <cstring>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
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
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema ContiguousBackward::schema_v1{"contiguous", 1, AmpPolicy::KeepInput, true};

TensorImplPtr ContiguousBackward::forward(const TensorImplPtr& a) {
    Validator::input(a, "contiguous.a").non_null();
    // No contiguous guard here — that's the whole point of this op.

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};

    Storage out_storage;
    if (a->device() == Device::GPU) {
        const auto& src = std::get<GpuStorage>(a->storage());
        if (!src.arr)
            ErrorBuilder("contiguous").fail("null GPU array");
        auto out = ::mlx::core::contiguous(*src.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype())};
    } else {
        const auto& src = std::get<CpuStorage>(a->storage());
        const std::size_t elem = dtype_size(a->dtype());
        const std::size_t n = shape_numel(a->shape());
        CpuStorage out;
        out.dtype = a->dtype();
        out.nbytes = n * elem;
        out.ptr = allocate_aligned_bytes(out.nbytes);
        if (out.nbytes > 0) {
            if (a->is_contiguous() && a->storage_offset() == 0) {
                std::memcpy(out.ptr.get(), src.ptr.get(), out.nbytes);
            } else {
                // Stride-walking copy: handles permuted/sliced views.
                const auto& shape = a->shape();
                const auto& stride = a->stride();
                const int ndim = static_cast<int>(shape.size());
                const std::uint8_t* base =
                    reinterpret_cast<const std::uint8_t*>(src.ptr.get()) + a->storage_offset();
                std::uint8_t* dst = reinterpret_cast<std::uint8_t*>(out.ptr.get());
                std::vector<std::size_t> coord(static_cast<std::size_t>(ndim), 0);
                for (std::size_t f = 0; f < n; ++f) {
                    std::ptrdiff_t byte_offset = 0;
                    for (int d = 0; d < ndim; ++d)
                        byte_offset += static_cast<std::ptrdiff_t>(coord[d]) *
                                       static_cast<std::ptrdiff_t>(stride[d]);
                    std::memcpy(dst + f * elem, base + byte_offset, elem);
                    for (int d = ndim - 1; d >= 0; --d) {
                        if (++coord[d] < static_cast<std::size_t>(shape[d]))
                            break;
                        coord[d] = 0;
                    }
                }
            }
        }
        out_storage = Storage{std::move(out)};
    }

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), a->shape(), a->dtype(),
                                               a->device(), false);

    if (!GradMode::is_enabled() || !a->requires_grad())
        return result;

    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<ContiguousBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = a->shape();
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    bwd->input_tensors_ = {a};
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, /*input_nr=*/0)});
    bwd->set_saved_versions({a->version()});

    result->set_grad_fn(std::move(bwd));
    result->set_leaf(false);
    result->set_requires_grad(true);
    return result;
}

std::vector<Storage> ContiguousBackward::apply(Storage grad_out) {
    // Identity backward: gradient passes through (cloned so engine owns it).
    return {clone_storage(grad_out, shape_numel(out_shape_), dtype_, device_)};
}

TensorImplPtr contiguous_op(const TensorImplPtr& a) {
    return ContiguousBackward::forward(a);
}

LUCID_REGISTER_OP(ContiguousBackward)

}  // namespace lucid
