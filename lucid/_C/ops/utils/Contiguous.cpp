#include "Contiguous.h"

#include <cstring>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema ContiguousBackward::schema_v1{
    "contiguous", 1, AmpPolicy::KeepInput, true};

TensorImplPtr ContiguousBackward::forward(const TensorImplPtr& a) {
    if (!a) throw LucidError("contiguous: null input");
    // No contiguous guard here — that's the whole point of this op.

    OpScope scope{schema_v1.name, a->device_, a->dtype_, a->shape_};

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        const auto& src = std::get<GpuStorage>(a->storage_);
        if (!src.arr) throw LucidError("contiguous: null GPU array");
        auto out = ::mlx::core::contiguous(*src.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        // Phase 3.4 v1: all CPU tensors are already contiguous, so this is just
        // a safe clone. Future zero-copy views can swap this for stride walking.
        const auto& src = std::get<CpuStorage>(a->storage_);
        CpuStorage out;
        out.dtype  = a->dtype_;
        out.nbytes = src.nbytes;
        out.ptr    = allocate_aligned_bytes(out.nbytes);
        if (out.nbytes > 0)
            std::memcpy(out.ptr.get(), src.ptr.get(), out.nbytes);
        out_storage = Storage{std::move(out)};
    }

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), a->shape_,
                                               a->dtype_, a->device_, false);

    if (!GradMode::is_enabled() || !a->requires_grad_) return result;

    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<ContiguousBackward>();
    bwd->input_shapes_  = {a->shape_};
    bwd->out_shape_     = a->shape_;
    bwd->dtype_         = a->dtype_;
    bwd->device_        = a->device_;
    bwd->input_tensors_ = {a};
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, /*input_nr=*/0)});
    bwd->set_saved_versions({a->version_});

    result->grad_fn_       = std::move(bwd);
    result->is_leaf_       = false;
    result->requires_grad_ = true;
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
