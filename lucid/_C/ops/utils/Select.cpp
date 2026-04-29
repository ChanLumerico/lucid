#include "Select.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/AutogradNode.h"
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
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::numel;
using utils_detail::wrap_axis;

template <typename T>
CpuStorage where_branch_cpu(const CpuStorage& grad,
                            const CpuStorage& cond,
                            const Shape& shape,
                            Dtype dt,
                            bool true_branch) {
    auto out = allocate_cpu(shape, dt);
    const std::size_t n = numel(shape);
    const auto* g = reinterpret_cast<const T*>(grad.ptr.get());
    const auto* c = reinterpret_cast<const std::uint8_t*>(cond.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.ptr.get());
    for (std::size_t i = 0; i < n; ++i) {
        const bool take = c[i] != 0;
        dst[i] = (take == true_branch) ? g[i] : T{};
    }
    return out;
}

Storage where_branch_storage(const Storage& grad,
                             const Storage& cond,
                             const Shape& shape,
                             Dtype dt,
                             Device device,
                             bool true_branch) {
    if (device == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad);
        const auto& gc = std::get<GpuStorage>(cond);
        auto zero = ::mlx::core::zeros(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
        auto out = true_branch ? ::mlx::core::where(*gc.arr, *gg.arr, zero)
                               : ::mlx::core::where(*gc.arr, zero, *gg.arr);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& g = std::get<CpuStorage>(grad);
    const auto& c = std::get<CpuStorage>(cond);
    switch (dt) {
        case Dtype::F32:
            return Storage{where_branch_cpu<float>(g, c, shape, dt, true_branch)};
        case Dtype::F64:
            return Storage{where_branch_cpu<double>(g, c, shape, dt, true_branch)};
        default:
            ErrorBuilder("where backward").not_implemented("dtype not supported");
    }
}

template <typename T>
CpuStorage gather_backward_cpu_typed(const CpuStorage& grad,
                                     const CpuStorage& indices,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     int axis,
                                     Dtype index_dtype,
                                     Dtype dt) {
    auto out = allocate_cpu(input_shape, dt);
    const std::size_t ndim = input_shape.size();
    Stride input_stride(ndim), output_stride(ndim);
    if (ndim > 0) {
        input_stride.back() = 1;
        output_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            input_stride[static_cast<std::size_t>(d)] =
                input_stride[static_cast<std::size_t>(d) + 1] *
                input_shape[static_cast<std::size_t>(d) + 1];
            output_stride[static_cast<std::size_t>(d)] =
                output_stride[static_cast<std::size_t>(d) + 1] *
                output_shape[static_cast<std::size_t>(d) + 1];
        }
    }
    auto load_idx = [&](std::size_t flat) -> std::int64_t {
        if (index_dtype == Dtype::I32)
            return reinterpret_cast<const std::int32_t*>(indices.ptr.get())[flat];
        if (index_dtype == Dtype::I64)
            return reinterpret_cast<const std::int64_t*>(indices.ptr.get())[flat];
        ErrorBuilder("gather backward").not_implemented("indices dtype must be I32 or I64");
    };

    const auto* g = reinterpret_cast<const T*>(grad.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.ptr.get());
    const std::size_t total = numel(output_shape);
    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
        std::int64_t k = load_idx(out_flat);
        if (k < 0)
            k += input_shape[static_cast<std::size_t>(axis)];
        std::size_t input_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::int64_t c = (static_cast<int>(d) == axis) ? k : coord[d];
            input_flat += static_cast<std::size_t>(c) * static_cast<std::size_t>(input_stride[d]);
        }
        dst[input_flat] += g[out_flat];
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < output_shape[static_cast<std::size_t>(d)]) {
                break;
            }
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return out;
}

Storage gather_backward_storage(const Storage& grad,
                                const Storage& indices,
                                const Shape& input_shape,
                                const Shape& output_shape,
                                int axis,
                                Dtype index_dtype,
                                Dtype dt,
                                Device device) {
    if (device == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad);
        const auto& gi = std::get<GpuStorage>(indices);
        auto idx = *gi.arr;
        auto axis_len = ::mlx::core::array(
            static_cast<std::int32_t>(input_shape[static_cast<std::size_t>(axis)]), idx.dtype());
        auto zero = ::mlx::core::array(static_cast<std::int32_t>(0), idx.dtype());
        auto fixed =
            ::mlx::core::where(::mlx::core::less(idx, zero), ::mlx::core::add(idx, axis_len), idx);
        auto base = ::mlx::core::zeros(gpu::to_mlx_shape(input_shape), gpu::to_mlx_dtype(dt));
        auto out = ::mlx::core::scatter_add_axis(base, fixed, *gg.arr, axis);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& g = std::get<CpuStorage>(grad);
    const auto& idx = std::get<CpuStorage>(indices);
    switch (dt) {
        case Dtype::F32:
            return Storage{gather_backward_cpu_typed<float>(g, idx, input_shape, output_shape, axis,
                                                            index_dtype, dt)};
        case Dtype::F64:
            return Storage{gather_backward_cpu_typed<double>(g, idx, input_shape, output_shape,
                                                             axis, index_dtype, dt)};
        default:
            ErrorBuilder("gather backward").not_implemented("dtype not supported");
    }
}

template <typename T>
CpuStorage diagonal_backward_cpu_typed(const CpuStorage& grad,
                                       const Shape& input_shape,
                                       const Shape& output_shape,
                                       int offset,
                                       int axis1,
                                       int axis2,
                                       Dtype dt) {
    auto out = allocate_cpu(input_shape, dt);
    const std::size_t ndim = input_shape.size();
    int a1 = axis1;
    int a2 = axis2;
    if (a1 > a2)
        std::swap(a1, a2);

    const std::int64_t M = input_shape[static_cast<std::size_t>(a1)];
    const std::int64_t N = input_shape[static_cast<std::size_t>(a2)];
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

    Stride input_stride(ndim);
    if (ndim > 0) {
        input_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            input_stride[static_cast<std::size_t>(d)] =
                input_stride[static_cast<std::size_t>(d) + 1] *
                input_shape[static_cast<std::size_t>(d) + 1];
        }
    }

    std::vector<std::size_t> outer_dims;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (static_cast<int>(d) != a1 && static_cast<int>(d) != a2)
            outer_dims.push_back(d);
    }
    std::size_t outer_numel = 1;
    for (auto d : outer_dims)
        outer_numel *= static_cast<std::size_t>(input_shape[d]);

    const auto* g = reinterpret_cast<const T*>(grad.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.ptr.get());
    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t o = 0; o < outer_numel; ++o) {
        std::size_t rem = o;
        for (auto d : outer_dims) {
            std::size_t prod = 1;
            for (std::size_t e : outer_dims)
                if (e > d)
                    prod *= static_cast<std::size_t>(input_shape[e]);
            coord[d] = static_cast<std::int64_t>(rem / prod);
            rem %= prod;
        }
        for (std::int64_t i = 0; i < L; ++i) {
            coord[static_cast<std::size_t>(a1)] = r0 + i;
            coord[static_cast<std::size_t>(a2)] = c0 + i;
            std::size_t input_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d)
                input_flat +=
                    static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(input_stride[d]);
            const std::size_t grad_flat =
                o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
            dst[input_flat] += g[grad_flat];
        }
    }
    (void)output_shape;
    return out;
}

Storage diagonal_backward_storage(const Storage& grad,
                                  const Shape& input_shape,
                                  const Shape& output_shape,
                                  int offset,
                                  int axis1,
                                  int axis2,
                                  Dtype dt,
                                  Device device) {
    if (device == Device::GPU) {
        // Native MLX path: scatter_add along all axes simultaneously.
        // Construct one index array per input dim sized to grad's shape,
        // selecting the diagonal positions for axis1/axis2 and pass-through
        // positions for batch axes.
        const auto& gg = std::get<GpuStorage>(grad);
        const std::size_t ndim = input_shape.size();
        const int a1n = axis1 < 0 ? axis1 + static_cast<int>(ndim) : axis1;
        const int a2n = axis2 < 0 ? axis2 + static_cast<int>(ndim) : axis2;
        const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
        const std::int64_t c0 = (offset >= 0) ? offset : 0;
        const std::int64_t L = output_shape.empty() ? 0 : output_shape.back();

        ::mlx::core::Shape mlx_in_shape = gpu::to_mlx_shape(input_shape);
        ::mlx::core::Shape mlx_out_shape = gpu::to_mlx_shape(output_shape);

        auto base = ::mlx::core::zeros(mlx_in_shape, gpu::to_mlx_dtype(dt));

        // Helper: construct an int32 index array of shape `mlx_out_shape`
        // whose values along the index_axis equal `arange(start, start+L)`,
        // and along all other axes are broadcast from a singleton.
        auto build_index = [&](int axis_in_input, std::int64_t start) {
            // Position of this axis within output_shape:
            //   batch axes (excluding axis1, axis2) keep their original
            //   relative order in output_shape; axis1/axis2 collapse into
            //   the trailing L axis.
            int out_axis;
            if (axis_in_input == a1n || axis_in_input == a2n) {
                out_axis = static_cast<int>(output_shape.size()) - 1;
            } else {
                int rel = 0;
                for (int d = 0; d < axis_in_input; ++d)
                    if (d != a1n && d != a2n)
                        ++rel;
                out_axis = rel;
            }
            const std::int64_t span = (axis_in_input == a1n || axis_in_input == a2n)
                                          ? L
                                          : input_shape[static_cast<std::size_t>(axis_in_input)];
            auto arr = ::mlx::core::arange(static_cast<int>(start), static_cast<int>(start + span),
                                           ::mlx::core::int32);
            // Reshape to broadcast along out_axis.
            ::mlx::core::Shape bc(output_shape.size(), 1);
            bc[static_cast<std::size_t>(out_axis)] = static_cast<int>(span);
            arr = ::mlx::core::reshape(arr, bc);
            return ::mlx::core::broadcast_to(arr, mlx_out_shape);
        };

        std::vector<::mlx::core::array> indices;
        std::vector<int> axes_v;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::int64_t start = 0;
            if (static_cast<int>(d) == a1n)
                start = r0;
            else if (static_cast<int>(d) == a2n)
                start = c0;
            indices.push_back(build_index(static_cast<int>(d), start));
            axes_v.push_back(static_cast<int>(d));
        }
        // MLX scatter contract: updates.shape = indices.shape + (1,)*ndim_a
        // when axes covers every dim of `a` (each scatter-position writes a
        // single element). Reshape grad so each entry sits in its own
        // (1,...,1) trailing slice.
        ::mlx::core::Shape upd_shape = mlx_out_shape;
        for (std::size_t d = 0; d < ndim; ++d)
            upd_shape.push_back(1);
        auto updates = ::mlx::core::reshape(*gg.arr, upd_shape);
        auto out = ::mlx::core::scatter_add(base, indices, updates, axes_v);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }

    const auto& g = std::get<CpuStorage>(grad);
    CpuStorage out;
    switch (dt) {
        case Dtype::F32:
            out = diagonal_backward_cpu_typed<float>(g, input_shape, output_shape, offset, axis1,
                                                     axis2, dt);
            break;
        case Dtype::F64:
            out = diagonal_backward_cpu_typed<double>(g, input_shape, output_shape, offset, axis1,
                                                      axis2, dt);
            break;
        default:
            ErrorBuilder("diagonal backward").not_implemented("dtype not supported");
    }
    return Storage{std::move(out)};
}

class WhereBackward : public AutogradNode<WhereBackward, 2> {
public:
    static const OpSchema schema_v1;

    Storage cond_;
    Shape shape_;
    std::weak_ptr<TensorImpl> cond_tensor_;
    std::weak_ptr<TensorImpl> x_tensor_;
    std::weak_ptr<TensorImpl> y_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {where_branch_storage(grad_out, cond_, shape_, dtype_, device_, true),
                where_branch_storage(grad_out, cond_, shape_, dtype_, device_, false)};
    }

    void validate_versions() override {
        check_version_match(cond_tensor_, saved_versions_.size() > 0 ? saved_versions_[0] : 0,
                            schema_v1.name, 0);
        check_version_match(x_tensor_, saved_versions_.size() > 1 ? saved_versions_[1] : 0,
                            schema_v1.name, 1);
        check_version_match(y_tensor_, saved_versions_.size() > 2 ? saved_versions_[2] : 0,
                            schema_v1.name, 2);
    }
};

const OpSchema WhereBackward::schema_v1{"where", 1, AmpPolicy::KeepInput, true};

class MaskedFillBackward : public AutogradNode<MaskedFillBackward, 1> {
public:
    static const OpSchema schema_v1;

    Storage mask_;
    Shape shape_;
    std::weak_ptr<TensorImpl> input_tensor_;
    std::weak_ptr<TensorImpl> mask_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {where_branch_storage(grad_out, mask_, shape_, dtype_, device_, false)};
    }

    void validate_versions() override {
        check_version_match(input_tensor_, saved_versions_.size() > 0 ? saved_versions_[0] : 0,
                            schema_v1.name, 0);
        check_version_match(mask_tensor_, saved_versions_.size() > 1 ? saved_versions_[1] : 0,
                            schema_v1.name, 1);
    }
};

const OpSchema MaskedFillBackward::schema_v1{"masked_fill", 1, AmpPolicy::KeepInput, true};

class RollBackward : public FuncOp<RollBackward, 1> {
public:
    static const OpSchema schema_v1;

    std::vector<std::int64_t> shifts_;
    std::vector<int> axes_;

    std::vector<Storage> apply(Storage grad_out) override {
        std::vector<std::int64_t> inv_shifts;
        inv_shifts.reserve(shifts_.size());
        for (auto s : shifts_)
            inv_shifts.push_back(-s);
        if (device_ == Device::GPU) {
            const auto& g = std::get<GpuStorage>(grad_out);
            ::mlx::core::Shape mshifts(inv_shifts.begin(), inv_shifts.end());
            auto out = ::mlx::core::roll(*g.arr, mshifts, axes_);
            return {Storage{gpu::wrap_mlx_array(std::move(out), dtype_)}};
        }
        auto t =
            std::make_shared<TensorImpl>(std::move(grad_out), out_shape_, dtype_, device_, false);
        auto out = roll_op(t, std::move(inv_shifts), axes_);
        return {out->storage()};
    }
};

const OpSchema RollBackward::schema_v1{"roll", 1, AmpPolicy::KeepInput, true};

class GatherBackward : public AutogradNode<GatherBackward, 1> {
public:
    static const OpSchema schema_v1;

    Storage indices_;
    Shape input_shape_;
    Shape output_shape_;
    int axis_ = 0;
    Dtype index_dtype_ = Dtype::I64;
    std::weak_ptr<TensorImpl> input_tensor_;
    std::weak_ptr<TensorImpl> indices_tensor_;

    std::vector<Storage> apply(Storage grad_out) override {
        return {gather_backward_storage(grad_out, indices_, input_shape_, output_shape_, axis_,
                                        index_dtype_, dtype_, device_)};
    }

    void validate_versions() override {
        check_version_match(input_tensor_, saved_versions_.size() > 0 ? saved_versions_[0] : 0,
                            schema_v1.name, 0);
        check_version_match(indices_tensor_, saved_versions_.size() > 1 ? saved_versions_[1] : 0,
                            schema_v1.name, 1);
    }
};

const OpSchema GatherBackward::schema_v1{"gather", 1, AmpPolicy::KeepInput, true};

class DiagonalBackward : public FuncOp<DiagonalBackward, 1> {
public:
    static const OpSchema schema_v1;

    int offset_ = 0;
    int axis1_ = 0;
    int axis2_ = 1;

    std::vector<Storage> apply(Storage grad_out) override {
        return {diagonal_backward_storage(grad_out, input_shapes_[0], out_shape_, offset_, axis1_,
                                          axis2_, dtype_, device_)};
    }
};

const OpSchema DiagonalBackward::schema_v1{"diagonal", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_where_grad(const TensorImplPtr& cond,
                                const TensorImplPtr& x,
                                const TensorImplPtr& y,
                                TensorImplPtr out) {
    const bool needs_grad = GradMode::is_enabled() && (x->requires_grad() || y->requires_grad());
    if (!needs_grad)
        return out;

    auto bwd = std::make_shared<WhereBackward>();
    bwd->cond_ = cond->storage();
    bwd->shape_ = out->shape();
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->cond_tensor_ = cond;
    bwd->x_tensor_ = x;
    bwd->y_tensor_ = y;
    bwd->set_next_edges(
        std::vector<Edge>{Edge(detail::ensure_grad_fn(x), 0), Edge(detail::ensure_grad_fn(y), 0)});
    bwd->set_saved_versions({cond->version(), x->version(), y->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr attach_masked_fill_grad(const TensorImplPtr& a,
                                      const TensorImplPtr& mask,
                                      TensorImplPtr out) {
    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;

    auto bwd = std::make_shared<MaskedFillBackward>();
    bwd->mask_ = mask->storage();
    bwd->shape_ = out->shape();
    bwd->dtype_ = out->dtype();
    bwd->device_ = out->device();
    bwd->input_tensor_ = a;
    bwd->mask_tensor_ = mask;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version(), mask->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr attach_unary_grad(const TensorImplPtr& a,
                                TensorImplPtr out,
                                std::shared_ptr<Node> bwd) {
    if (!GradMode::is_enabled() || !a->requires_grad())
        return out;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version()});
    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

LUCID_REGISTER_OP(WhereBackward)
LUCID_REGISTER_OP(MaskedFillBackward)
LUCID_REGISTER_OP(RollBackward)
LUCID_REGISTER_OP(GatherBackward)
LUCID_REGISTER_OP(DiagonalBackward)

}  // namespace

TensorImplPtr where_op(const TensorImplPtr& cond, const TensorImplPtr& x, const TensorImplPtr& y) {
    if (!cond || !x || !y)
        ErrorBuilder("where").fail("null input");
    if (x->dtype() != y->dtype())
        throw DtypeMismatch(std::string(dtype_name(x->dtype())),
                            std::string(dtype_name(y->dtype())), "where");
    if (x->device() != y->device() || cond->device() != x->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(y->device())), "where");
    const Dtype dt = x->dtype();
    const Device device = x->device();
    OpScopeFull scope{"where", device, dt, x->shape()};
    if (device == Device::GPU) {
        const auto& gc = std::get<GpuStorage>(cond->storage());
        const auto& gx = std::get<GpuStorage>(x->storage());
        const auto& gy = std::get<GpuStorage>(y->storage());
        auto out = ::mlx::core::where(*gc.arr, *gx.arr, *gy.arr);
        Shape sh = mlx_shape_to_lucid(out.shape());
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(sh), dt, device);
        return attach_where_grad(cond, x, y, std::move(result));
    }
    if (cond->shape() != x->shape() || x->shape() != y->shape())
        throw ShapeMismatch(x->shape(), y->shape(), "where (CPU same-shape)");
    auto out_cpu = allocate_cpu(x->shape(), dt);
    const std::size_t n = numel(x->shape());
    const auto* c =
        reinterpret_cast<const std::uint8_t*>(std::get<CpuStorage>(cond->storage()).ptr.get());
    auto run = [&](auto* dst, const auto* xp, const auto* yp) {
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = c[i] ? xp[i] : yp[i];
    };
    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(out_cpu.ptr.get()),
            reinterpret_cast<const float*>(std::get<CpuStorage>(x->storage()).ptr.get()),
            reinterpret_cast<const float*>(std::get<CpuStorage>(y->storage()).ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(out_cpu.ptr.get()),
            reinterpret_cast<const double*>(std::get<CpuStorage>(x->storage()).ptr.get()),
            reinterpret_cast<const double*>(std::get<CpuStorage>(y->storage()).ptr.get()));
    else
        ErrorBuilder("where").not_implemented("dtype not supported");
    auto result = fresh(Storage{std::move(out_cpu)}, x->shape(), dt, device);
    return attach_where_grad(cond, x, y, std::move(result));
}

TensorImplPtr masked_fill_op(const TensorImplPtr& a, const TensorImplPtr& mask, double value) {
    if (!a || !mask)
        ErrorBuilder("masked_fill").fail("null input");
    if (a->shape() != mask->shape())
        throw ShapeMismatch(a->shape(), mask->shape(), "masked_fill");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"masked_fill", device, dt, a->shape()};
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        const auto& gm = std::get<GpuStorage>(mask->storage());
        ::mlx::core::array v(static_cast<float>(value), gpu::to_mlx_dtype(dt));
        auto out = ::mlx::core::where(*gm.arr, v, *ga.arr);
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, a->shape(), dt, device);
        return attach_masked_fill_grad(a, mask, std::move(result));
    }
    auto out_cpu = allocate_cpu(a->shape(), dt);
    const std::size_t n = numel(a->shape());
    const auto* m =
        reinterpret_cast<const std::uint8_t*>(std::get<CpuStorage>(mask->storage()).ptr.get());
    auto run = [&](auto* dst, const auto* src) {
        using T = std::remove_pointer_t<decltype(dst)>;
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = m[i] ? static_cast<T>(value) : src[i];
    };
    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(out_cpu.ptr.get()),
            reinterpret_cast<const float*>(std::get<CpuStorage>(a->storage()).ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(out_cpu.ptr.get()),
            reinterpret_cast<const double*>(std::get<CpuStorage>(a->storage()).ptr.get()));
    else
        ErrorBuilder("masked_fill").not_implemented("dtype not supported");
    auto result = fresh(Storage{std::move(out_cpu)}, a->shape(), dt, device);
    return attach_masked_fill_grad(a, mask, std::move(result));
}

TensorImplPtr roll_op(const TensorImplPtr& a,
                      std::vector<std::int64_t> shifts,
                      std::vector<int> axes) {
    Validator::input(a, "roll.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"roll", device, dt, a->shape()};
    if (shifts.size() != axes.size())
        ErrorBuilder("roll").fail("shifts and axes must have equal length");
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        ::mlx::core::Shape mshifts(shifts.begin(), shifts.end());
        auto out = ::mlx::core::roll(*ga.arr, mshifts, axes);
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, a->shape(), dt, device);
        auto bwd = std::make_shared<RollBackward>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = result->shape();
        bwd->dtype_ = dt;
        bwd->device_ = device;
        bwd->input_tensors_ = {a};
        bwd->shifts_ = shifts;
        bwd->axes_ = axes;
        return attach_unary_grad(a, std::move(result), std::move(bwd));
    }
    const std::size_t ndim = a->shape().size();
    std::vector<std::int64_t> shift_per_dim(ndim, 0);
    for (std::size_t i = 0; i < axes.size(); ++i) {
        const int ax = wrap_axis(axes[i], static_cast<int>(ndim));
        shift_per_dim[ax] += shifts[i];
    }
    Shape out_shape = a->shape();
    auto out_cpu = allocate_cpu(out_shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage());
    const std::size_t elem = dtype_size(dt);
    Stride stride(ndim);
    if (ndim > 0) {
        stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 2; d >= 0; --d)
            stride[d] = stride[d + 1] * out_shape[d + 1];
    }
    const std::size_t total = numel(out_shape);
    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
        std::size_t in_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::int64_t c = coord[d] - shift_per_dim[d];
            std::int64_t L = out_shape[d];
            c = ((c % L) + L) % L;
            in_flat += static_cast<std::size_t>(c) * static_cast<std::size_t>(stride[d]);
        }
        std::memcpy(out_cpu.ptr.get() + out_flat * elem, ca.ptr.get() + in_flat * elem, elem);
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 1; d >= 0; --d) {
            if (++coord[d] < out_shape[d])
                break;
            coord[d] = 0;
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    auto bwd = std::make_shared<RollBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->shifts_ = std::move(shifts);
    bwd->axes_ = std::move(axes);
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

TensorImplPtr gather_op(const TensorImplPtr& a, const TensorImplPtr& indices, int axis) {
    if (!a || !indices)
        ErrorBuilder("gather").fail("null input");
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"gather", device, dt, indices->shape()};
    if (a->shape().size() != indices->shape().size())
        throw ShapeMismatch(a->shape(), indices->shape(),
                            "gather: a and indices must have same rank");
    const std::size_t ndim = a->shape().size();
    int ax = wrap_axis(axis, static_cast<int>(ndim));
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        const auto& gi = std::get<GpuStorage>(indices->storage());
        auto out = ::mlx::core::take_along_axis(*ga.arr, *gi.arr, ax);
        Shape sh = mlx_shape_to_lucid(out.shape());
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(sh), dt, device);
        if (GradMode::is_enabled() && a->requires_grad()) {
            auto bwd = std::make_shared<GatherBackward>();
            bwd->indices_ = indices->storage();
            bwd->input_shape_ = a->shape();
            bwd->output_shape_ = result->shape();
            bwd->axis_ = ax;
            bwd->dtype_ = dt;
            bwd->index_dtype_ = indices->dtype();
            bwd->device_ = device;
            bwd->input_tensor_ = a;
            bwd->indices_tensor_ = indices;
            bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
            bwd->set_saved_versions({a->version(), indices->version()});
            result->set_grad_fn(std::move(bwd));
            result->set_leaf(false);
            result->set_requires_grad(true);
        }
        return result;
    }
    Shape out_shape = indices->shape();
    auto out_cpu = allocate_cpu(out_shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage());
    const auto& ci = std::get<CpuStorage>(indices->storage());
    const std::size_t elem = dtype_size(dt);

    Stride a_stride(ndim), out_stride(ndim);
    if (ndim > 0) {
        a_stride.back() = 1;
        out_stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 2; d >= 0; --d) {
            a_stride[d] = a_stride[d + 1] * a->shape()[d + 1];
            out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
        }
    }
    const std::size_t total = numel(out_shape);

    auto load_idx = [&](std::size_t flat) -> std::int64_t {
        if (indices->dtype() == Dtype::I32)
            return reinterpret_cast<const std::int32_t*>(ci.ptr.get())[flat];
        if (indices->dtype() == Dtype::I64)
            return reinterpret_cast<const std::int64_t*>(ci.ptr.get())[flat];
        ErrorBuilder("gather").not_implemented("indices dtype must be I32 or I64");
    };

    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
        std::int64_t k = load_idx(out_flat);
        if (k < 0)
            k += a->shape()[ax];
        std::size_t a_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::int64_t c = (static_cast<int>(d) == ax) ? k : coord[d];
            a_flat += static_cast<std::size_t>(c) * static_cast<std::size_t>(a_stride[d]);
        }
        std::memcpy(out_cpu.ptr.get() + out_flat * elem, ca.ptr.get() + a_flat * elem, elem);
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 1; d >= 0; --d) {
            if (++coord[d] < out_shape[d])
                break;
            coord[d] = 0;
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    if (GradMode::is_enabled() && a->requires_grad()) {
        auto bwd = std::make_shared<GatherBackward>();
        bwd->indices_ = indices->storage();
        bwd->input_shape_ = a->shape();
        bwd->output_shape_ = result->shape();
        bwd->axis_ = ax;
        bwd->dtype_ = dt;
        bwd->index_dtype_ = indices->dtype();
        bwd->device_ = device;
        bwd->input_tensor_ = a;
        bwd->indices_tensor_ = indices;
        bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
        bwd->set_saved_versions({a->version(), indices->version()});
        result->set_grad_fn(std::move(bwd));
        result->set_leaf(false);
        result->set_requires_grad(true);
    }
    return result;
}

TensorImplPtr diagonal_op(const TensorImplPtr& a, int offset, int axis1, int axis2) {
    Validator::input(a, "diagonal.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"diagonal", device, dt, a->shape()};
    const std::size_t ndim = a->shape().size();
    if (ndim < 2)
        ErrorBuilder("diagonal").fail("input must be ≥2-D");
    int a1 = wrap_axis(axis1, static_cast<int>(ndim));
    int a2 = wrap_axis(axis2, static_cast<int>(ndim));
    if (a1 == a2)
        ErrorBuilder("diagonal").fail("axis1 and axis2 must differ");
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        // MLX diagonal returns a strided view; contiguous() materializes a
        // dense buffer matching lucid's row-major layout convention.
        auto out = ::mlx::core::contiguous(::mlx::core::diagonal(*ga.arr, offset, a1, a2));
        Shape sh = mlx_shape_to_lucid(out.shape());
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(sh), dt, device);
        auto bwd = std::make_shared<DiagonalBackward>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = result->shape();
        bwd->dtype_ = dt;
        bwd->device_ = device;
        bwd->input_tensors_ = {a};
        bwd->offset_ = offset;
        bwd->axis1_ = a1;
        bwd->axis2_ = a2;
        return attach_unary_grad(a, std::move(result), std::move(bwd));
    }
    if (a1 > a2)
        std::swap(a1, a2);

    const std::int64_t M = a->shape()[a1];
    const std::int64_t N = a->shape()[a2];
    const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
    const std::int64_t c0 = (offset >= 0) ? offset : 0;
    const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

    Shape out_shape;
    for (std::size_t d = 0; d < ndim; ++d) {
        if ((int)d == a1 || (int)d == a2)
            continue;
        out_shape.push_back(a->shape()[d]);
    }
    out_shape.push_back(L);
    auto out_cpu = allocate_cpu(out_shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage());
    const std::size_t elem = dtype_size(dt);

    Stride a_stride(ndim);
    if (ndim > 0) {
        a_stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 2; d >= 0; --d)
            a_stride[d] = a_stride[d + 1] * a->shape()[d + 1];
    }

    std::vector<std::size_t> outer_dims;
    for (std::size_t d = 0; d < ndim; ++d)
        if ((int)d != a1 && (int)d != a2)
            outer_dims.push_back(d);

    std::size_t outer_numel = 1;
    for (auto d : outer_dims)
        outer_numel *= static_cast<std::size_t>(a->shape()[d]);

    std::vector<std::int64_t> coord(ndim, 0);
    for (std::size_t o = 0; o < outer_numel; ++o) {
        std::size_t rem = o;
        for (auto d : outer_dims) {
            std::size_t prod = 1;
            for (std::size_t e : outer_dims)
                if (e > d)
                    prod *= static_cast<std::size_t>(a->shape()[e]);
            coord[d] = rem / prod;
            rem %= prod;
        }
        for (std::int64_t i = 0; i < L; ++i) {
            coord[a1] = r0 + i;
            coord[a2] = c0 + i;
            std::size_t a_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d)
                a_flat +=
                    static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(a_stride[d]);
            const std::size_t out_flat =
                o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
            std::memcpy(out_cpu.ptr.get() + out_flat * elem, ca.ptr.get() + a_flat * elem, elem);
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    auto bwd = std::make_shared<DiagonalBackward>();
    bwd->input_shapes_ = {a->shape()};
    bwd->out_shape_ = result->shape();
    bwd->dtype_ = dt;
    bwd->device_ = device;
    bwd->input_tensors_ = {a};
    bwd->offset_ = offset;
    bwd->axis1_ = a1;
    bwd->axis2_ = a2;
    return attach_unary_grad(a, std::move(result), std::move(bwd));
}

}  // namespace lucid
