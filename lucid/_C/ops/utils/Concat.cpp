#include "Concat.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::check_dtype_device_match;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::wrap_axis;

CpuStorage slice_axis_cpu(const CpuStorage& src, const Shape& src_shape,
                          Shape slice_shape, int axis, std::int64_t offset,
                          Dtype dt) {
    auto out = allocate_cpu(slice_shape, dt);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(src_shape[d]);
    std::size_t inner_bytes = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1;
         d < src_shape.size(); ++d) {
        inner_bytes *= static_cast<std::size_t>(src_shape[d]);
    }
    const std::size_t src_axis = static_cast<std::size_t>(src_shape[axis]);
    const std::size_t slice_axis =
        static_cast<std::size_t>(slice_shape[axis]);
    const std::size_t copy_bytes = slice_axis * inner_bytes;
    for (std::size_t o = 0; o < outer; ++o) {
        const auto* src_ptr =
            src.ptr.get() + (o * src_axis + static_cast<std::size_t>(offset)) *
                                inner_bytes;
        auto* dst_ptr = out.ptr.get() + o * copy_bytes;
        if (copy_bytes > 0) std::memcpy(dst_ptr, src_ptr, copy_bytes);
    }
    return out;
}

CpuStorage insert_axis_slice_cpu(const CpuStorage& src, const Shape& src_shape,
                                 const Shape& dst_shape, int axis,
                                 std::int64_t offset, Dtype dt) {
    auto out = allocate_cpu(dst_shape, dt);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(dst_shape[d]);
    std::size_t inner_bytes = elem;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1;
         d < dst_shape.size(); ++d) {
        inner_bytes *= static_cast<std::size_t>(dst_shape[d]);
    }
    const std::size_t dst_axis = static_cast<std::size_t>(dst_shape[axis]);
    const std::size_t src_axis = static_cast<std::size_t>(src_shape[axis]);
    const std::size_t copy_bytes = src_axis * inner_bytes;
    for (std::size_t o = 0; o < outer; ++o) {
        const auto* src_ptr = src.ptr.get() + o * copy_bytes;
        auto* dst_ptr =
            out.ptr.get() + (o * dst_axis + static_cast<std::size_t>(offset)) *
                                inner_bytes;
        if (copy_bytes > 0) std::memcpy(dst_ptr, src_ptr, copy_bytes);
    }
    return out;
}

Storage slice_axis_storage(const Storage& src, const Shape& src_shape,
                           const Shape& slice_shape, int axis,
                           std::int64_t offset, Dtype dt, Device device) {
    if (device == Device::GPU) {
        const auto& g = std::get<GpuStorage>(src);
        ::mlx::core::Shape lo(src_shape.size(), 0);
        ::mlx::core::Shape hi = gpu::to_mlx_shape(src_shape);
        lo[static_cast<std::size_t>(axis)] =
            static_cast<::mlx::core::ShapeElem>(offset);
        hi[static_cast<std::size_t>(axis)] =
            static_cast<::mlx::core::ShapeElem>(offset + slice_shape[axis]);
        auto out = ::mlx::core::slice(*g.arr, lo, hi);
        out = ::mlx::core::contiguous(out);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    return Storage{slice_axis_cpu(std::get<CpuStorage>(src), src_shape,
                                  slice_shape, axis, offset, dt)};
}

Storage insert_axis_slice_storage(const Storage& src, const Shape& src_shape,
                                  const Shape& dst_shape, int axis,
                                  std::int64_t offset, Dtype dt,
                                  Device device) {
    if (device == Device::GPU) {
        const auto& g = std::get<GpuStorage>(src);
        std::vector<std::pair<int, int>> pad;
        pad.reserve(dst_shape.size());
        for (std::size_t d = 0; d < dst_shape.size(); ++d) {
            if (static_cast<int>(d) == axis) {
                const auto before = static_cast<int>(offset);
                const auto after = static_cast<int>(
                    dst_shape[d] - offset - src_shape[d]);
                pad.emplace_back(before, after);
            } else {
                pad.emplace_back(0, 0);
            }
        }
        ::mlx::core::array zero(static_cast<float>(0.0),
                                gpu::to_mlx_dtype(dt));
        auto out = ::mlx::core::pad(*g.arr, pad, zero);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    return Storage{insert_axis_slice_cpu(std::get<CpuStorage>(src), src_shape,
                                         dst_shape, axis, offset, dt)};
}

class ConcatBackward : public Node {
public:
    static const OpSchema schema_v1;

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_;
    std::vector<Shape> input_shapes_;
    Shape output_shape_;
    int axis_ = 0;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    std::vector<Storage> apply(Storage grad_out) override {
        std::vector<Storage> grads;
        grads.reserve(input_shapes_.size());
        std::int64_t offset = 0;
        for (const auto& shape : input_shapes_) {
            grads.push_back(slice_axis_storage(grad_out, output_shape_, shape,
                                               axis_, offset, dtype_, device_));
            offset += shape[static_cast<std::size_t>(axis_)];
        }
        return grads;
    }

    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            check_version_match(input_tensors_[i],
                                saved_versions_.size() > i
                                    ? saved_versions_[i]
                                    : 0,
                                schema_v1.name, i);
        }
    }
};

const OpSchema ConcatBackward::schema_v1{
    "concatenate", 1, AmpPolicy::KeepInput, true};

class StackBackward : public Node {
public:
    static const OpSchema schema_v1;

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_;
    Shape input_shape_;
    Shape output_shape_;
    int axis_ = 0;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    std::vector<Storage> apply(Storage grad_out) override {
        Shape slice_shape = output_shape_;
        slice_shape[static_cast<std::size_t>(axis_)] = 1;
        std::vector<Storage> grads;
        grads.reserve(input_tensors_.size());
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            auto piece = slice_axis_storage(
                grad_out, output_shape_, slice_shape, axis_,
                static_cast<std::int64_t>(i), dtype_, device_);
            if (device_ == Device::GPU) {
                auto& g = std::get<GpuStorage>(piece);
                auto reshaped =
                    ::mlx::core::reshape(*g.arr, gpu::to_mlx_shape(input_shape_));
                grads.push_back(
                    Storage{gpu::wrap_mlx_array(std::move(reshaped), dtype_)});
            } else {
                auto& c = std::get<CpuStorage>(piece);
                CpuStorage out;
                out.dtype = dtype_;
                out.nbytes = c.nbytes;
                out.ptr = allocate_aligned_bytes(out.nbytes);
                if (out.nbytes > 0) {
                    std::memcpy(out.ptr.get(), c.ptr.get(), out.nbytes);
                }
                grads.push_back(Storage{std::move(out)});
            }
        }
        return grads;
    }

    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_.size(); ++i) {
            check_version_match(input_tensors_[i],
                                saved_versions_.size() > i
                                    ? saved_versions_[i]
                                    : 0,
                                schema_v1.name, i);
        }
    }
};

const OpSchema StackBackward::schema_v1{
    "stack", 1, AmpPolicy::KeepInput, true};

class SplitSliceBackward : public FuncOp<SplitSliceBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape slice_shape_;
    int axis_ = 0;
    std::int64_t offset_ = 0;
    bool squeeze_axis_ = false;

    std::vector<Storage> apply(Storage grad_out) override {
        Storage slice_grad = std::move(grad_out);
        if (squeeze_axis_) {
            if (device_ == Device::GPU) {
                auto& g = std::get<GpuStorage>(slice_grad);
                auto reshaped =
                    ::mlx::core::reshape(*g.arr, gpu::to_mlx_shape(slice_shape_));
                slice_grad =
                    Storage{gpu::wrap_mlx_array(std::move(reshaped), dtype_)};
            } else {
                const auto& c = std::get<CpuStorage>(slice_grad);
                CpuStorage out;
                out.dtype = dtype_;
                out.nbytes = c.nbytes;
                out.ptr = allocate_aligned_bytes(out.nbytes);
                if (out.nbytes > 0) {
                    std::memcpy(out.ptr.get(), c.ptr.get(), out.nbytes);
                }
                slice_grad = Storage{std::move(out)};
            }
        }
        return {insert_axis_slice_storage(slice_grad, slice_shape_,
                                          input_shapes_[0], axis_, offset_,
                                          dtype_, device_)};
    }
};

const OpSchema SplitSliceBackward::schema_v1{
    "split", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_concat_grad(const std::vector<TensorImplPtr>& xs,
                                 TensorImplPtr out, int axis) {
    bool needs_grad = GradMode::is_enabled();
    if (needs_grad) {
        needs_grad = false;
        for (const auto& t : xs) needs_grad = needs_grad || t->requires_grad_;
    }
    if (!needs_grad) return out;

    auto bwd = std::make_shared<ConcatBackward>();
    bwd->input_shapes_.reserve(xs.size());
    bwd->input_tensors_.reserve(xs.size());
    bwd->output_shape_ = out->shape_;
    bwd->axis_ = axis;
    bwd->dtype_ = out->dtype_;
    bwd->device_ = out->device_;

    std::vector<Edge> edges;
    std::vector<std::int64_t> versions;
    edges.reserve(xs.size());
    versions.reserve(xs.size());
    for (const auto& t : xs) {
        bwd->input_shapes_.push_back(t->shape_);
        bwd->input_tensors_.push_back(t);
        edges.emplace_back(detail::ensure_grad_fn(t), 0);
        versions.push_back(t->version_);
    }
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions(std::move(versions));

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

TensorImplPtr attach_stack_grad(const std::vector<TensorImplPtr>& xs,
                                TensorImplPtr out, int axis) {
    bool needs_grad = GradMode::is_enabled();
    if (needs_grad) {
        needs_grad = false;
        for (const auto& t : xs) needs_grad = needs_grad || t->requires_grad_;
    }
    if (!needs_grad) return out;

    auto bwd = std::make_shared<StackBackward>();
    bwd->input_shape_ = xs[0]->shape_;
    bwd->output_shape_ = out->shape_;
    bwd->axis_ = axis;
    bwd->dtype_ = out->dtype_;
    bwd->device_ = out->device_;
    bwd->input_tensors_.reserve(xs.size());

    std::vector<Edge> edges;
    std::vector<std::int64_t> versions;
    edges.reserve(xs.size());
    versions.reserve(xs.size());
    for (const auto& t : xs) {
        bwd->input_tensors_.push_back(t);
        edges.emplace_back(detail::ensure_grad_fn(t), 0);
        versions.push_back(t->version_);
    }
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions(std::move(versions));

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

TensorImplPtr attach_split_grad(const TensorImplPtr& a, TensorImplPtr out,
                                Shape slice_shape, int axis,
                                std::int64_t offset, bool squeeze_axis) {
    if (!GradMode::is_enabled() || !a->requires_grad_) return out;

    auto bwd = std::make_shared<SplitSliceBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->slice_shape_ = std::move(slice_shape);
    bwd->axis_ = axis;
    bwd->offset_ = offset;
    bwd->squeeze_axis_ = squeeze_axis;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

LUCID_REGISTER_OP(ConcatBackward)
LUCID_REGISTER_OP(StackBackward)
LUCID_REGISTER_OP(SplitSliceBackward)

}  // namespace

TensorImplPtr concatenate_op(const std::vector<TensorImplPtr>& xs, int axis) {
    check_dtype_device_match(xs, "concatenate");
    const Dtype dt = xs[0]->dtype_;
    const Device device = xs[0]->device_;
    OpScope scope{"concatenate", device, dt, Shape{}};
    const auto ndim = xs[0]->shape_.size();
    int ax = wrap_axis(axis, static_cast<int>(ndim));

    if (device == Device::GPU) {
        std::vector<::mlx::core::array> arrays;
        arrays.reserve(xs.size());
        for (auto& t : xs) arrays.push_back(*std::get<GpuStorage>(t->storage_).arr);
        auto out = ::mlx::core::concatenate(std::move(arrays), ax);
        Shape out_shape = mlx_shape_to_lucid(out.shape());
        auto result = fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)},
                            std::move(out_shape), dt, device);
        return attach_concat_grad(xs, std::move(result), ax);
    }

    Shape out_shape = xs[0]->shape_;
    std::int64_t cat_dim = 0;
    for (auto& t : xs) {
        if (t->shape_.size() != ndim)
            throw ShapeMismatch(xs[0]->shape_, t->shape_, "concatenate");
        for (std::size_t d = 0; d < ndim; ++d) {
            if ((int)d == ax) continue;
            if (t->shape_[d] != xs[0]->shape_[d])
                throw ShapeMismatch(xs[0]->shape_, t->shape_, "concatenate");
        }
        cat_dim += t->shape_[ax];
    }
    out_shape[ax] = cat_dim;

    auto out_cpu = allocate_cpu(out_shape, dt);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= static_cast<std::size_t>(out_shape[d]);
    std::size_t inner_per_unit = elem;
    for (std::size_t d = ax + 1; d < ndim; ++d)
        inner_per_unit *= static_cast<std::size_t>(out_shape[d]);
    auto* dst = out_cpu.ptr.get();
    for (std::size_t o = 0; o < outer; ++o) {
        for (auto& t : xs) {
            const auto& cs = std::get<CpuStorage>(t->storage_);
            const std::size_t L = static_cast<std::size_t>(t->shape_[ax]);
            const std::size_t bytes = L * inner_per_unit;
            std::memcpy(dst, cs.ptr.get() + o * bytes, bytes);
            dst += bytes;
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt,
                        device);
    return attach_concat_grad(xs, std::move(result), ax);
}

TensorImplPtr stack_op(const std::vector<TensorImplPtr>& xs, int axis) {
    check_dtype_device_match(xs, "stack");
    const Dtype dt = xs[0]->dtype_;
    const Device device = xs[0]->device_;
    OpScope scope{"stack", device, dt, Shape{}};
    const auto ndim = xs[0]->shape_.size();
    int ax = axis < 0 ? axis + static_cast<int>(ndim) + 1 : axis;
    if (ax < 0 || ax > static_cast<int>(ndim)) {
        throw IndexError("stack: axis out of range");
    }
    for (const auto& t : xs) {
        if (t->shape_ != xs[0]->shape_) {
            throw ShapeMismatch(xs[0]->shape_, t->shape_, "stack");
        }
    }

    if (device == Device::GPU) {
        std::vector<::mlx::core::array> arrays;
        arrays.reserve(xs.size());
        for (auto& t : xs) arrays.push_back(*std::get<GpuStorage>(t->storage_).arr);
        auto out = ::mlx::core::stack(arrays, ax);
        Shape out_shape = mlx_shape_to_lucid(out.shape());
        auto result = fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)},
                            std::move(out_shape), dt, device);
        return attach_stack_grad(xs, std::move(result), ax);
    }

    Shape out_shape = xs[0]->shape_;
    out_shape.insert(out_shape.begin() + ax, static_cast<std::int64_t>(xs.size()));
    auto out_cpu = allocate_cpu(out_shape, dt);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= static_cast<std::size_t>(out_shape[d]);
    std::size_t inner_bytes = elem;
    for (std::size_t d = static_cast<std::size_t>(ax) + 1;
         d < out_shape.size(); ++d) {
        inner_bytes *= static_cast<std::size_t>(out_shape[d]);
    }
    const std::size_t block_bytes =
        static_cast<std::size_t>(xs.size()) * inner_bytes;
    for (std::size_t idx = 0; idx < xs.size(); ++idx) {
        const auto& cs = std::get<CpuStorage>(xs[idx]->storage_);
        for (std::size_t o = 0; o < outer; ++o) {
            std::memcpy(out_cpu.ptr.get() + o * block_bytes + idx * inner_bytes,
                        cs.ptr.get() + o * inner_bytes, inner_bytes);
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt,
                        device);
    return attach_stack_grad(xs, std::move(result), ax);
}

TensorImplPtr hstack_op(const std::vector<TensorImplPtr>& xs) {
    if (xs.empty()) throw LucidError("hstack: empty input");
    return concatenate_op(xs, xs[0]->shape_.size() <= 1 ? 0 : 1);
}

TensorImplPtr vstack_op(const std::vector<TensorImplPtr>& xs) {
    if (xs.empty()) throw LucidError("vstack: empty input");
    if (xs[0]->shape_.size() == 1) {
        return stack_op(xs, 0);
    }
    return concatenate_op(xs, 0);
}

std::vector<TensorImplPtr>
split_op(const TensorImplPtr& a, std::int64_t num_splits, int axis) {
    if (!a) throw LucidError("split: null input");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"split", device, dt, a->shape_};
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    if (num_splits <= 0)
        throw LucidError("split: num_splits must be positive");
    if (a->shape_[ax] % num_splits != 0)
        throw LucidError("split: dimension not divisible by num_splits");
    const std::int64_t piece = a->shape_[ax] / num_splits;

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto pieces = ::mlx::core::split(*ga.arr,
                                         static_cast<int>(num_splits), ax);
        std::vector<TensorImplPtr> out;
        out.reserve(pieces.size());
        std::int64_t k = 0;
        for (auto& p : pieces) {
            // MLX split returns strided views; materialize so element layout
            // matches numpy/PyTorch when downloaded.
            auto pc = ::mlx::core::contiguous(p);
            Shape sh = mlx_shape_to_lucid(pc.shape());
            auto result =
                fresh(Storage{gpu::wrap_mlx_array(std::move(pc), dt)}, sh,
                      dt, device);
            out.push_back(attach_split_grad(a, std::move(result), std::move(sh),
                                            ax, k * piece,
                                            /*squeeze_axis=*/false));
            ++k;
        }
        return out;
    }

    std::vector<TensorImplPtr> out;
    out.reserve(num_splits);
    Shape piece_shape = a->shape_;
    piece_shape[ax] = piece;

    const auto& ca = std::get<CpuStorage>(a->storage_);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= static_cast<std::size_t>(a->shape_[d]);
    std::size_t inner_per_unit = elem;
    for (std::size_t d = ax + 1; d < a->shape_.size(); ++d)
        inner_per_unit *= static_cast<std::size_t>(a->shape_[d]);
    const std::size_t full_row_bytes =
        static_cast<std::size_t>(a->shape_[ax]) * inner_per_unit;
    const std::size_t piece_bytes =
        static_cast<std::size_t>(piece) * inner_per_unit;

    for (std::int64_t k = 0; k < num_splits; ++k) {
        auto cpu = allocate_cpu(piece_shape, dt);
        for (std::size_t o = 0; o < outer; ++o) {
            std::memcpy(cpu.ptr.get() + o * piece_bytes,
                        ca.ptr.get() + o * full_row_bytes
                            + static_cast<std::size_t>(k) * piece_bytes,
                        piece_bytes);
        }
        auto result = fresh(Storage{std::move(cpu)}, piece_shape, dt, device);
        out.push_back(attach_split_grad(a, std::move(result), piece_shape, ax,
                                        k * piece, /*squeeze_axis=*/false));
    }
    return out;
}

std::vector<TensorImplPtr>
split_at_op(const TensorImplPtr& a, std::vector<std::int64_t> indices,
            int axis) {
    if (!a) throw LucidError("split_at: null input");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"split_at", device, dt, a->shape_};
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        ::mlx::core::Shape mlx_idx(indices.begin(), indices.end());
        int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
        auto pieces = ::mlx::core::split(*ga.arr, mlx_idx, ax);
        std::vector<TensorImplPtr> out;
        std::int64_t off = 0;
        for (auto& p : pieces) {
            auto pc = ::mlx::core::contiguous(p);
            Shape sh = mlx_shape_to_lucid(pc.shape());
            auto result =
                fresh(Storage{gpu::wrap_mlx_array(std::move(pc), dt)}, sh,
                      dt, device);
            out.push_back(attach_split_grad(a, std::move(result), sh, ax, off,
                                            /*squeeze_axis=*/false));
            off += sh[static_cast<std::size_t>(ax)];
        }
        return out;
    }
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    std::vector<std::int64_t> sizes;
    sizes.reserve(indices.size() + 1);
    std::int64_t prev = 0;
    for (auto idx : indices) {
        sizes.push_back(idx - prev);
        prev = idx;
    }
    sizes.push_back(a->shape_[ax] - prev);
    const auto& ca = std::get<CpuStorage>(a->storage_);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d) outer *= static_cast<std::size_t>(a->shape_[d]);
    std::size_t inner_per_unit = elem;
    for (std::size_t d = ax + 1; d < a->shape_.size(); ++d)
        inner_per_unit *= static_cast<std::size_t>(a->shape_[d]);
    const std::size_t full_row_bytes =
        static_cast<std::size_t>(a->shape_[ax]) * inner_per_unit;

    std::vector<TensorImplPtr> out;
    out.reserve(sizes.size());
    std::int64_t off = 0;
    for (auto sz : sizes) {
        Shape piece_shape = a->shape_;
        piece_shape[ax] = sz;
        auto cpu = allocate_cpu(piece_shape, dt);
        const std::size_t piece_bytes =
            static_cast<std::size_t>(sz) * inner_per_unit;
        for (std::size_t o = 0; o < outer; ++o) {
            std::memcpy(cpu.ptr.get() + o * piece_bytes,
                        ca.ptr.get() + o * full_row_bytes
                            + static_cast<std::size_t>(off) * inner_per_unit,
                        piece_bytes);
        }
        auto result = fresh(Storage{std::move(cpu)}, piece_shape, dt, device);
        out.push_back(attach_split_grad(a, std::move(result), piece_shape, ax,
                                        off, /*squeeze_axis=*/false));
        off += sz;
    }
    return out;
}

std::vector<TensorImplPtr>
chunk_op(const TensorImplPtr& a, std::int64_t chunks, int axis) {
    return split_op(a, chunks, axis);
}

std::vector<TensorImplPtr>
unbind_op(const TensorImplPtr& a, int axis) {
    if (!a) throw LucidError("unbind: null input");
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    std::vector<TensorImplPtr> out;
    out.reserve(static_cast<std::size_t>(a->shape_[ax]));
    Shape slice_shape = a->shape_;
    slice_shape[static_cast<std::size_t>(ax)] = 1;
    Shape out_shape = slice_shape;
    out_shape.erase(out_shape.begin() + ax);

    for (std::int64_t k = 0; k < a->shape_[ax]; ++k) {
        auto piece = slice_axis_storage(a->storage_, a->shape_, slice_shape,
                                        ax, k, a->dtype_, a->device_);
        if (a->device_ == Device::GPU) {
            auto& g = std::get<GpuStorage>(piece);
            auto reshaped =
                ::mlx::core::reshape(*g.arr, gpu::to_mlx_shape(out_shape));
            auto result =
                fresh(Storage{gpu::wrap_mlx_array(std::move(reshaped),
                                                  a->dtype_)},
                      out_shape, a->dtype_, a->device_);
            out.push_back(attach_split_grad(a, std::move(result), slice_shape,
                                            ax, k, /*squeeze_axis=*/true));
        } else {
            auto result = fresh(std::move(piece), out_shape, a->dtype_,
                                a->device_);
            out.push_back(attach_split_grad(a, std::move(result), slice_shape,
                                            ax, k, /*squeeze_axis=*/true));
        }
    }
    return out;
}

}  // namespace lucid
