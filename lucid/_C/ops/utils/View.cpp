#include "View.h"

#include <cstring>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema ViewBackward::schema_v1{"view", 1, AmpPolicy::KeepInput, true};

namespace {

CpuStorage allocate_like(const Shape& shape, Dtype dt) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = shape_numel(shape) * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    return out;
}

// Copy raw bytes — same buffer, same layout (contiguous source, contiguous
// destination), only metadata changes. Used by reshape/squeeze/unsqueeze.
CpuStorage clone_for_view(const CpuStorage& src, const Shape& out_shape, Dtype dt) {
    auto out = allocate_like(out_shape, dt);
    if (out.nbytes > 0)
        std::memcpy(out.ptr.get(), src.ptr.get(), out.nbytes);
    return out;
}

// Handle a `reshape` shape with one optional `-1` placeholder.
Shape resolve_reshape_shape(const Shape& in_shape, const std::vector<std::int64_t>& new_shape) {
    const std::size_t in_numel = shape_numel(in_shape);
    Shape resolved;
    resolved.reserve(new_shape.size());
    int wildcard_pos = -1;
    std::int64_t known_product = 1;
    for (std::size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (wildcard_pos != -1) {
                ErrorBuilder("reshape").fail("only one -1 is allowed");
            }
            wildcard_pos = static_cast<int>(i);
            resolved.push_back(0);  // placeholder
        } else if (new_shape[i] < 0) {
            ErrorBuilder("reshape").fail("negative dim other than -1 is invalid");
        } else {
            known_product *= new_shape[i];
            resolved.push_back(new_shape[i]);
        }
    }
    if (wildcard_pos != -1) {
        if (known_product == 0) {
            ErrorBuilder("reshape").fail("cannot infer -1 from a product of zero");
        }
        if (in_numel % static_cast<std::size_t>(known_product) != 0) {
            throw ShapeMismatch(in_shape, resolved,
                                "reshape: -1 inference failed (numel mismatch)");
        }
        resolved[wildcard_pos] = static_cast<std::int64_t>(in_numel) / known_product;
    }
    if (shape_numel(resolved) != in_numel) {
        throw ShapeMismatch(in_shape, resolved, "reshape: total numel mismatch");
    }
    return resolved;
}

TensorImplPtr build_view_output(const TensorImplPtr& a, Shape out_shape, const char* op_name) {
    Validator::input(a, std::string(op_name) + ".a").non_null();
    if (a->device_ == Device::GPU) {
        // GPU: reshape via MLX, materialize via contiguous to avoid lazy-view
        // hazards downstream.
        OpScopeFull scope{op_name, a->device_, a->dtype_, out_shape};
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto out = ::mlx::core::reshape(*ga.arr, gpu::to_mlx_shape(out_shape));
        out = ::mlx::core::contiguous(out);
        auto out_t =
            std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)},
                                         out_shape, a->dtype_, a->device_, /*requires_grad=*/false);

        if (GradMode::is_enabled() && a->requires_grad_) {
            auto a_edge = detail::ensure_grad_fn(a);
            auto bwd = std::make_shared<ViewBackward>();
            bwd->input_shapes_ = {a->shape_};
            bwd->out_shape_ = out_t->shape_;
            bwd->dtype_ = a->dtype_;
            bwd->device_ = a->device_;
            bwd->input_tensors_ = {a};
            bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
            bwd->set_saved_versions({a->version_});
            out_t->grad_fn_ = std::move(bwd);
            out_t->is_leaf_ = false;
            out_t->requires_grad_ = true;
        }
        return out_t;
    }
    if (!a->is_contiguous()) {
        ErrorBuilder(op_name).not_implemented(
            "non-contiguous input not supported (call .contiguous() first)");
    }

    OpScopeFull scope{op_name, a->device_, a->dtype_, out_shape};
    auto out_cpu = clone_for_view(std::get<CpuStorage>(a->storage_), out_shape, a->dtype_);
    Storage out_storage{std::move(out_cpu)};

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype_, a->device_,
                                     /*requires_grad=*/false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<ViewBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, /*input_nr=*/0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

}  // namespace

std::vector<Storage> ViewBackward::apply(Storage grad_out) {
    // dx layout = input_shape (contiguous). For both CPU and GPU, the inverse
    // is just a reshape back to the original shape.
    if (device_ == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad_out);
        auto out = ::mlx::core::reshape(*gg.arr, gpu::to_mlx_shape(input_shapes_[0]));
        out = ::mlx::core::contiguous(out);
        return {Storage{gpu::wrap_mlx_array(std::move(out), dtype_)}};
    }
    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    auto out_cpu = clone_for_view(g_cpu, input_shapes_[0], dtype_);
    return {Storage{std::move(out_cpu)}};
}

// ---------------- reshape ----------------
TensorImplPtr reshape_op(const TensorImplPtr& a, const std::vector<std::int64_t>& new_shape) {
    Validator::input(a, "reshape.a").non_null();
    Shape resolved = resolve_reshape_shape(a->shape_, new_shape);
    return build_view_output(a, std::move(resolved), "reshape");
}

// ---------------- squeeze ----------------
TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim) {
    Validator::input(a, "squeeze.a").non_null();
    const int ndim = static_cast<int>(a->shape_.size());
    if (ndim == 0) {
        // Squeezing a 0-d tensor on an axis — error per PyTorch semantics.
        ErrorBuilder("squeeze").index_error("axis out of range for 0-d tensor");
    }
    const int wrapped = dim < 0 ? dim + ndim : dim;
    if (wrapped < 0 || wrapped >= ndim) {
        ErrorBuilder("squeeze").index_error("axis out of range");
    }
    if (a->shape_[static_cast<std::size_t>(wrapped)] != 1) {
        ErrorBuilder("squeeze").fail("target dim must be size 1");
    }
    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i != wrapped)
            new_shape.push_back(a->shape_[static_cast<std::size_t>(i)]);
    }
    return build_view_output(a, std::move(new_shape), "squeeze");
}

// ---------------- squeeze_all ----------------
TensorImplPtr squeeze_all_op(const TensorImplPtr& a) {
    Validator::input(a, "squeeze.a").non_null();
    Shape new_shape;
    for (auto d : a->shape_) {
        if (d != 1)
            new_shape.push_back(d);
    }
    return build_view_output(a, std::move(new_shape), "squeeze");
}

// ---------------- unsqueeze ----------------
TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim) {
    Validator::input(a, "unsqueeze.a").non_null();
    const int ndim = static_cast<int>(a->shape_.size());
    // unsqueeze allows dim in [-(ndim+1), ndim] — one beyond the end.
    const int wrapped = dim < 0 ? dim + ndim + 1 : dim;
    if (wrapped < 0 || wrapped > ndim) {
        ErrorBuilder("unsqueeze").index_error("axis out of range");
    }
    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) + 1);
    for (int i = 0; i <= ndim; ++i) {
        if (i == wrapped) {
            new_shape.push_back(1);
        }
        if (i < ndim) {
            new_shape.push_back(a->shape_[static_cast<std::size_t>(i)]);
        }
    }
    return build_view_output(a, std::move(new_shape), "unsqueeze");
}

LUCID_REGISTER_OP(ViewBackward)

}  // namespace lucid
