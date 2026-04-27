#include "Vision.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../backend/cpu/Blas.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype  = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr    = allocate_aligned_bytes(s.nbytes);
    return s;
}

inline std::int64_t read_index(const CpuStorage& ts, std::size_t i) {
    switch (ts.dtype) {
        case Dtype::I8:  return reinterpret_cast<const std::int8_t*>(ts.ptr.get())[i];
        case Dtype::I16: return reinterpret_cast<const std::int16_t*>(ts.ptr.get())[i];
        case Dtype::I32: return reinterpret_cast<const std::int32_t*>(ts.ptr.get())[i];
        case Dtype::I64: return reinterpret_cast<const std::int64_t*>(ts.ptr.get())[i];
        case Dtype::Bool: return reinterpret_cast<const std::uint8_t*>(ts.ptr.get())[i];
        default:
            throw NotImplementedError("one_hot/rotate: index dtype must be integer");
    }
}

}  // namespace

// =====================================================================
// one_hot — no autograd; integer-indexed scatter.
// =====================================================================

TensorImplPtr one_hot_op(const TensorImplPtr& input, int num_classes,
                           Dtype out_dtype) {
    if (!input) throw LucidError("one_hot: null input");
    if (num_classes <= 0)
        throw LucidError("one_hot: num_classes must be > 0");
    Shape out_shape = input->shape_;
    out_shape.push_back(num_classes);
    OpScope scope{"one_hot", input->device_, out_dtype, out_shape};

    if (input->device_ == Device::GPU) {
        const auto& gi = std::get<GpuStorage>(input->storage_);
        // arange [num_classes] vs broadcast input → equality.
        auto cls = ::mlx::core::astype(::mlx::core::arange(0, num_classes, 1),
                                          ::mlx::core::int64);
        auto idx = ::mlx::core::astype(*gi.arr, ::mlx::core::int64);
        // Insert trailing dim, broadcast against arange.
        ::mlx::core::Shape idx_shape = idx.shape();
        idx_shape.push_back(1);
        auto idx_b = ::mlx::core::reshape(idx, idx_shape);
        ::mlx::core::Shape cls_shape(idx_shape.size(), 1);
        cls_shape.back() = num_classes;
        auto cls_b = ::mlx::core::reshape(cls, cls_shape);
        auto eq = ::mlx::core::equal(idx_b, cls_b);
        auto mlx_dt = gpu::to_mlx_dtype(out_dtype);
        auto out = ::mlx::core::astype(eq, mlx_dt);
        return std::make_shared<TensorImpl>(
            Storage{gpu::wrap_mlx_array(std::move(out), out_dtype)},
            out_shape, out_dtype, input->device_, false);
    }
    const std::size_t M = input->numel();
    auto out_cpu = allocate_size(M * static_cast<std::size_t>(num_classes), out_dtype);
    std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);
    const auto& is_ = std::get<CpuStorage>(input->storage_);
    auto write_one = [&](std::size_t i, std::int64_t cls) {
        const std::size_t pos = i * static_cast<std::size_t>(num_classes) + cls;
        switch (out_dtype) {
            case Dtype::F32:
                reinterpret_cast<float*>(out_cpu.ptr.get())[pos] = 1.f; break;
            case Dtype::F64:
                reinterpret_cast<double*>(out_cpu.ptr.get())[pos] = 1.0; break;
            case Dtype::I8: case Dtype::Bool:
                reinterpret_cast<std::uint8_t*>(out_cpu.ptr.get())[pos] = 1; break;
            case Dtype::I16:
                reinterpret_cast<std::int16_t*>(out_cpu.ptr.get())[pos] = 1; break;
            case Dtype::I32:
                reinterpret_cast<std::int32_t*>(out_cpu.ptr.get())[pos] = 1; break;
            case Dtype::I64:
                reinterpret_cast<std::int64_t*>(out_cpu.ptr.get())[pos] = 1; break;
            default:
                throw NotImplementedError("one_hot: output dtype not supported");
        }
    };
    for (std::size_t i = 0; i < M; ++i) {
        std::int64_t cls = read_index(is_, i);
        if (cls < 0 || cls >= num_classes) continue;  // PyTorch silently skips
        write_one(i, cls);
    }
    return std::make_shared<TensorImpl>(Storage{std::move(out_cpu)},
                                         out_shape, out_dtype,
                                         input->device_, false);
}

// =====================================================================
// rotate — no autograd; nearest-neighbor sample with affine matrix.
// =====================================================================

namespace {

template <typename T>
void rotate_cpu_kernel(const T* xp, T* op, int N, int C, int H, int W,
                        double angle_rad_neg, double cx, double cy) {
    const double c = std::cos(angle_rad_neg);
    const double s = std::sin(angle_rad_neg);
    for (int n = 0; n < N; ++n)
    for (int ch = 0; ch < C; ++ch) {
        const T* base = xp + (n * C + ch) * H * W;
        T* obase = op + (n * C + ch) * H * W;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const double xs_d = c * (x - cx) - s * (y - cy) + cx;
                const double ys_d = s * (x - cx) + c * (y - cy) + cy;
                int xs = static_cast<int>(std::floor(xs_d + 0.5));
                int ys = static_cast<int>(std::floor(ys_d + 0.5));
                if (xs < 0 || xs >= W || ys < 0 || ys >= H) {
                    obase[y * W + x] = T{0};
                } else {
                    obase[y * W + x] = base[ys * W + xs];
                }
            }
        }
    }
}

}  // namespace

TensorImplPtr rotate_op(const TensorImplPtr& input, double angle_deg,
                          double cy, double cx) {
    if (!input) throw LucidError("rotate: null input");
    if (input->shape_.size() != 4)
        throw ShapeMismatch(input->shape_, Shape{},
                             "rotate: input must be 4-D (N, C, H, W)");
    const int N = static_cast<int>(input->shape_[0]);
    const int C = static_cast<int>(input->shape_[1]);
    const int H = static_cast<int>(input->shape_[2]);
    const int W = static_cast<int>(input->shape_[3]);
    OpScope scope{"rotate", input->device_, input->dtype_, input->shape_};
    const double angle_rad_neg = -angle_deg * (M_PI / 180.0);

    if (input->device_ == Device::GPU) {
        // GPU: download → CPU compute → upload (nearest-neighbor remap is small).
        auto x_cpu = gpu::download_gpu_to_cpu(
            std::get<GpuStorage>(input->storage_), input->shape_);
        auto out_cpu = allocate_size(static_cast<std::size_t>(N) * C * H * W,
                                       input->dtype_);
        if (input->dtype_ == Dtype::F32)
            rotate_cpu_kernel<float>(
                reinterpret_cast<const float*>(x_cpu.ptr.get()),
                reinterpret_cast<float*>(out_cpu.ptr.get()),
                N, C, H, W, angle_rad_neg, cx, cy);
        else if (input->dtype_ == Dtype::F64)
            rotate_cpu_kernel<double>(
                reinterpret_cast<const double*>(x_cpu.ptr.get()),
                reinterpret_cast<double*>(out_cpu.ptr.get()),
                N, C, H, W, angle_rad_neg, cx, cy);
        else
            throw NotImplementedError("rotate: dtype must be F32/F64");
        return std::make_shared<TensorImpl>(
            Storage{gpu::upload_cpu_to_gpu(out_cpu, input->shape_)},
            input->shape_, input->dtype_, input->device_, false);
    }
    auto out_cpu = allocate_size(static_cast<std::size_t>(N) * C * H * W,
                                   input->dtype_);
    const auto& xs = std::get<CpuStorage>(input->storage_);
    if (input->dtype_ == Dtype::F32)
        rotate_cpu_kernel<float>(
            reinterpret_cast<const float*>(xs.ptr.get()),
            reinterpret_cast<float*>(out_cpu.ptr.get()),
            N, C, H, W, angle_rad_neg, cx, cy);
    else if (input->dtype_ == Dtype::F64)
        rotate_cpu_kernel<double>(
            reinterpret_cast<const double*>(xs.ptr.get()),
            reinterpret_cast<double*>(out_cpu.ptr.get()),
            N, C, H, W, angle_rad_neg, cx, cy);
    else
        throw NotImplementedError("rotate: dtype must be F32/F64");
    return std::make_shared<TensorImpl>(Storage{std::move(out_cpu)},
                                         input->shape_, input->dtype_,
                                         input->device_, false);
}

// =====================================================================
// bilinear (learned bilinear layer): y[..., k] = x1 W_k x2 + b_k
// =====================================================================
//
// x1: [..., D1]   x2: [..., D2]   weight: [Dout, D1, D2]   bias: [Dout]
// Forward:
//   tmp = einsum("...i, k i j -> ...k j", x1, W)            # [..., Dout, D2]
//   y   = einsum("...k j, ...j -> ...k", tmp, x2) + bias    # [..., Dout]
// Backward (broadcast over batch dims; we collapse leading dims to a single B):
//   dx1[k]  = sum_k dY[..., k] · W[k, :, :] · x2
//   dx2[j]  = sum_k dY[..., k] · W[k, i, :]^T · x1[i] (per-element); explicit form below.
//   dW[k,i,j] = sum_batch dY[..., k] · x1[i] · x2[j]
//   db[k]   = sum_batch dY[..., k]

const OpSchema BilinearLayerBackward::schema_v1{
    "bilinear_layer", 1, AmpPolicy::Promote, true};

namespace {

struct BilinearShape {
    std::size_t B;
    std::size_t D1;
    std::size_t D2;
    std::size_t Dout;
};

BilinearShape flatten_bilinear(const Shape& s1, const Shape& s2,
                                const Shape& sw) {
    if (s1.empty() || s2.empty()) {
        throw LucidError("bilinear: input must have ≥1 dim");
    }
    if (sw.size() != 3) {
        throw ShapeMismatch(sw, Shape{}, "bilinear: weight must be 3-D");
    }
    if (s1.size() != s2.size()) {
        throw ShapeMismatch(s1, s2, "bilinear: x1/x2 must have same rank");
    }
    BilinearShape r;
    r.Dout = static_cast<std::size_t>(sw[0]);
    r.D1   = static_cast<std::size_t>(sw[1]);
    r.D2   = static_cast<std::size_t>(sw[2]);
    if (static_cast<std::size_t>(s1.back()) != r.D1
        || static_cast<std::size_t>(s2.back()) != r.D2) {
        throw ShapeMismatch(s1, sw, "bilinear: last dims must match weight");
    }
    std::size_t b = 1;
    for (std::size_t i = 0; i + 1 < s1.size(); ++i) {
        if (s1[i] != s2[i])
            throw ShapeMismatch(s1, s2, "bilinear: batch dims must match");
        b *= static_cast<std::size_t>(s1[i]);
    }
    r.B = b;
    return r;
}

template <typename T>
void bilinear_forward_cpu(const T* x1, const T* x2, const T* W, const T* b,
                            T* y, std::size_t B, std::size_t D1, std::size_t D2,
                            std::size_t Dout) {
    for (std::size_t bi = 0; bi < B; ++bi) {
        const T* x1b = x1 + bi * D1;
        const T* x2b = x2 + bi * D2;
        T* yb = y + bi * Dout;
        for (std::size_t k = 0; k < Dout; ++k) {
            const T* Wk = W + k * D1 * D2;
            T acc = T{0};
            for (std::size_t i = 0; i < D1; ++i) {
                T row_acc = T{0};
                for (std::size_t j = 0; j < D2; ++j) {
                    row_acc += Wk[i * D2 + j] * x2b[j];
                }
                acc += x1b[i] * row_acc;
            }
            yb[k] = acc + (b ? b[k] : T{0});
        }
    }
}

template <typename T>
void bilinear_backward_cpu(const T* x1, const T* x2, const T* W, const T* gy,
                             T* dx1, T* dx2, T* dW, T* db,
                             std::size_t B, std::size_t D1, std::size_t D2,
                             std::size_t Dout, bool need_db) {
    std::memset(dx1, 0, sizeof(T) * B * D1);
    std::memset(dx2, 0, sizeof(T) * B * D2);
    std::memset(dW,  0, sizeof(T) * Dout * D1 * D2);
    if (db) std::memset(db, 0, sizeof(T) * Dout);
    (void)need_db;
    for (std::size_t bi = 0; bi < B; ++bi) {
        const T* x1b = x1 + bi * D1;
        const T* x2b = x2 + bi * D2;
        const T* gb = gy + bi * Dout;
        T* dx1b = dx1 + bi * D1;
        T* dx2b = dx2 + bi * D2;
        for (std::size_t k = 0; k < Dout; ++k) {
            const T g = gb[k];
            if (db) db[k] += g;
            const T* Wk = W + k * D1 * D2;
            T* dWk = dW + k * D1 * D2;
            for (std::size_t i = 0; i < D1; ++i) {
                const T x1_i = x1b[i];
                T row_acc = T{0};
                for (std::size_t j = 0; j < D2; ++j) {
                    const T w_ij = Wk[i * D2 + j];
                    row_acc += w_ij * x2b[j];
                    dx2b[j] += g * x1_i * w_ij;
                    dWk[i * D2 + j] += g * x1_i * x2b[j];
                }
                dx1b[i] += g * row_acc;
            }
        }
    }
}

}  // namespace

TensorImplPtr BilinearLayerBackward::forward(const TensorImplPtr& x1,
                                              const TensorImplPtr& x2,
                                              const TensorImplPtr& weight,
                                              const TensorImplPtr& bias) {
    if (!x1 || !x2 || !weight)
        throw LucidError("bilinear_layer: null input");
    if (x1->dtype_ != x2->dtype_ || x1->dtype_ != weight->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x1->dtype_)),
                            std::string(dtype_name(x2->dtype_)),
                            "bilinear_layer");
    if (x1->device_ != x2->device_ || x1->device_ != weight->device_)
        throw DeviceMismatch(std::string(device_name(x1->device_)),
                              std::string(device_name(x2->device_)),
                              "bilinear_layer");
    const auto bs = flatten_bilinear(x1->shape_, x2->shape_, weight->shape_);
    Shape out_shape = x1->shape_;
    out_shape.back() = static_cast<std::int64_t>(bs.Dout);
    OpScope scope{schema_v1.name, x1->device_, x1->dtype_, out_shape};

    Storage out_storage;
    if (x1->device_ == Device::GPU) {
        const auto& gx1 = std::get<GpuStorage>(x1->storage_);
        const auto& gx2 = std::get<GpuStorage>(x2->storage_);
        const auto& gw  = std::get<GpuStorage>(weight->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(x1->dtype_);
        // Reshape inputs: x1 → [B, D1], x2 → [B, D2], W → [Dout, D1, D2].
        auto x1_b = ::mlx::core::reshape(*gx1.arr,
            {static_cast<int>(bs.B), static_cast<int>(bs.D1)});
        auto x2_b = ::mlx::core::reshape(*gx2.arr,
            {static_cast<int>(bs.B), static_cast<int>(bs.D2)});
        // tmp[b, k, j] = sum_i x1[b, i] * W[k, i, j]
        // Using batched matmul: treat W as [Dout, D1, D2] → for each k a [D1, D2].
        // Easier path: tmp[b, k, j] = (x1[b, :] @ W[k, :, :])[j]
        // → tmp = einsum("bi, kij -> bkj"). Implement via:
        //   x1_e[b, 1, i] @ W[k, i, j] → broadcast not directly supported in mlx::matmul.
        // Use: x1_b [B, D1] @ W reshaped to [D1, Dout*D2] → [B, Dout*D2] → [B, Dout, D2].
        auto W_rs = ::mlx::core::transpose(*gw.arr, {1, 0, 2});  // [D1, Dout, D2]
        auto W_rs2 = ::mlx::core::reshape(W_rs,
            {static_cast<int>(bs.D1), static_cast<int>(bs.Dout * bs.D2)});
        auto tmp = ::mlx::core::matmul(x1_b, W_rs2);  // [B, Dout*D2]
        auto tmp_3d = ::mlx::core::reshape(tmp,
            {static_cast<int>(bs.B), static_cast<int>(bs.Dout),
              static_cast<int>(bs.D2)});
        // y[b, k] = sum_j tmp[b, k, j] * x2[b, j]
        auto x2_e = ::mlx::core::reshape(x2_b,
            {static_cast<int>(bs.B), static_cast<int>(bs.D2), 1});
        auto y_e = ::mlx::core::matmul(tmp_3d, x2_e);  // [B, Dout, 1]
        auto y_2d = ::mlx::core::reshape(y_e,
            {static_cast<int>(bs.B), static_cast<int>(bs.Dout)});
        if (bias) {
            const auto& gb = std::get<GpuStorage>(bias->storage_);
            y_2d = ::mlx::core::add(y_2d, *gb.arr);
        }
        auto y_out = ::mlx::core::reshape(y_2d, gpu::to_mlx_shape(out_shape));
        out_storage = Storage{gpu::wrap_mlx_array(std::move(y_out), x1->dtype_)};
    } else {
        auto out_cpu = allocate_size(bs.B * bs.Dout, x1->dtype_);
        const auto& xs1 = std::get<CpuStorage>(x1->storage_);
        const auto& xs2 = std::get<CpuStorage>(x2->storage_);
        const auto& ws  = std::get<CpuStorage>(weight->storage_);
        const CpuStorage* bs_ptr = bias ? &std::get<CpuStorage>(bias->storage_) : nullptr;
        if (x1->dtype_ == Dtype::F32)
            bilinear_forward_cpu<float>(
                reinterpret_cast<const float*>(xs1.ptr.get()),
                reinterpret_cast<const float*>(xs2.ptr.get()),
                reinterpret_cast<const float*>(ws.ptr.get()),
                bs_ptr ? reinterpret_cast<const float*>(bs_ptr->ptr.get()) : nullptr,
                reinterpret_cast<float*>(out_cpu.ptr.get()),
                bs.B, bs.D1, bs.D2, bs.Dout);
        else if (x1->dtype_ == Dtype::F64)
            bilinear_forward_cpu<double>(
                reinterpret_cast<const double*>(xs1.ptr.get()),
                reinterpret_cast<const double*>(xs2.ptr.get()),
                reinterpret_cast<const double*>(ws.ptr.get()),
                bs_ptr ? reinterpret_cast<const double*>(bs_ptr->ptr.get()) : nullptr,
                reinterpret_cast<double*>(out_cpu.ptr.get()),
                bs.B, bs.D1, bs.D2, bs.Dout);
        else
            throw NotImplementedError("bilinear_layer: dtype must be F32/F64");
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                             out_shape, x1->dtype_,
                                             x1->device_, false);
    const bool any_grad = x1->requires_grad_ || x2->requires_grad_
                          || weight->requires_grad_
                          || (bias && bias->requires_grad_);
    if (!GradMode::is_enabled() || !any_grad) return out;

    auto x1_edge = detail::ensure_grad_fn(x1);
    auto x2_edge = detail::ensure_grad_fn(x2);
    auto w_edge  = detail::ensure_grad_fn(weight);
    auto b_edge  = bias ? detail::ensure_grad_fn(bias) : std::shared_ptr<Node>();
    auto bwd = std::make_shared<BilinearLayerBackward>();
    if (bias) {
        bwd->input_shapes_ = {x1->shape_, x2->shape_, weight->shape_, bias->shape_};
        bwd->input_tensors_ = {x1, x2, weight, bias};
        bwd->saved_inputs_ = {x1->storage_, x2->storage_, weight->storage_,
                                bias->storage_};
    } else {
        // Bias slot is required by FuncOp<4>; store an empty CpuStorage and
        // a null TensorImplPtr — apply() will detect the empty slot and skip
        // the db output.
        bwd->input_shapes_ = {x1->shape_, x2->shape_, weight->shape_, Shape{}};
        bwd->input_tensors_[0] = x1;
        bwd->input_tensors_[1] = x2;
        bwd->input_tensors_[2] = weight;
        // Slot 3 left default-constructed (empty weak_ptr).
        CpuStorage empty;
        empty.dtype = x1->dtype_; empty.nbytes = 0;
        bwd->saved_inputs_ = {x1->storage_, x2->storage_, weight->storage_,
                                Storage{std::move(empty)}};
    }
    bwd->out_shape_ = out_shape;
    bwd->dtype_     = x1->dtype_;
    bwd->device_    = x1->device_;
    bwd->orig_x1_shape_ = x1->shape_;
    bwd->orig_x2_shape_ = x2->shape_;
    std::vector<Edge> edges{Edge(x1_edge, 0), Edge(x2_edge, 0), Edge(w_edge, 0)};
    if (bias) edges.emplace_back(b_edge, 0);
    else      edges.emplace_back(std::shared_ptr<Node>(), 0);
    bwd->set_next_edges(std::move(edges));
    std::vector<std::int64_t> versions{
        static_cast<std::int64_t>(x1->version_),
        static_cast<std::int64_t>(x2->version_),
        static_cast<std::int64_t>(weight->version_)};
    if (bias) versions.push_back(static_cast<std::int64_t>(bias->version_));
    else      versions.push_back(0);
    bwd->set_saved_versions(std::move(versions));
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> BilinearLayerBackward::apply(Storage grad_out) {
    const auto bs = flatten_bilinear(orig_x1_shape_, orig_x2_shape_,
                                       input_shapes_[2]);
    const bool has_bias = !input_shapes_[3].empty();

    // Bridge through CPU for both devices to keep autograd analytic + simple.
    CpuStorage x1_cpu, x2_cpu, w_cpu, b_cpu, gy_cpu;
    const CpuStorage* x1_ptr;
    const CpuStorage* x2_ptr;
    const CpuStorage* w_ptr;
    const CpuStorage* b_ptr;
    const CpuStorage* gy_ptr;
    if (device_ == Device::GPU) {
        x1_cpu = gpu::download_gpu_to_cpu(std::get<GpuStorage>(saved_inputs_[0]),
                                            orig_x1_shape_);
        x2_cpu = gpu::download_gpu_to_cpu(std::get<GpuStorage>(saved_inputs_[1]),
                                            orig_x2_shape_);
        w_cpu  = gpu::download_gpu_to_cpu(std::get<GpuStorage>(saved_inputs_[2]),
                                            input_shapes_[2]);
        if (has_bias)
            b_cpu = gpu::download_gpu_to_cpu(
                std::get<GpuStorage>(saved_inputs_[3]), input_shapes_[3]);
        gy_cpu = gpu::download_gpu_to_cpu(std::get<GpuStorage>(grad_out),
                                            out_shape_);
        x1_ptr = &x1_cpu; x2_ptr = &x2_cpu; w_ptr = &w_cpu;
        b_ptr = has_bias ? &b_cpu : nullptr;
        gy_ptr = &gy_cpu;
    } else {
        x1_ptr = &std::get<CpuStorage>(saved_inputs_[0]);
        x2_ptr = &std::get<CpuStorage>(saved_inputs_[1]);
        w_ptr  = &std::get<CpuStorage>(saved_inputs_[2]);
        b_ptr  = has_bias ? &std::get<CpuStorage>(saved_inputs_[3]) : nullptr;
        gy_ptr = &std::get<CpuStorage>(grad_out);
    }

    auto dx1_cpu = allocate_size(bs.B * bs.D1, dtype_);
    auto dx2_cpu = allocate_size(bs.B * bs.D2, dtype_);
    auto dW_cpu  = allocate_size(bs.Dout * bs.D1 * bs.D2, dtype_);
    auto db_cpu  = has_bias ? allocate_size(bs.Dout, dtype_) : CpuStorage{};

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        bilinear_backward_cpu<T>(
            reinterpret_cast<const T*>(x1_ptr->ptr.get()),
            reinterpret_cast<const T*>(x2_ptr->ptr.get()),
            reinterpret_cast<const T*>(w_ptr->ptr.get()),
            reinterpret_cast<const T*>(gy_ptr->ptr.get()),
            reinterpret_cast<T*>(dx1_cpu.ptr.get()),
            reinterpret_cast<T*>(dx2_cpu.ptr.get()),
            reinterpret_cast<T*>(dW_cpu.ptr.get()),
            has_bias ? reinterpret_cast<T*>(db_cpu.ptr.get()) : nullptr,
            bs.B, bs.D1, bs.D2, bs.Dout, has_bias);
    };
    if (dtype_ == Dtype::F32) run(float{});
    else if (dtype_ == Dtype::F64) run(double{});
    else throw NotImplementedError("bilinear_layer backward: dtype not supported");

    if (device_ == Device::GPU) {
        std::vector<Storage> out_grads;
        out_grads.push_back(Storage{gpu::upload_cpu_to_gpu(dx1_cpu, orig_x1_shape_)});
        out_grads.push_back(Storage{gpu::upload_cpu_to_gpu(dx2_cpu, orig_x2_shape_)});
        out_grads.push_back(Storage{gpu::upload_cpu_to_gpu(dW_cpu, input_shapes_[2])});
        if (has_bias)
            out_grads.push_back(Storage{
                gpu::upload_cpu_to_gpu(db_cpu, input_shapes_[3])});
        else {
            CpuStorage empty;
            empty.dtype = dtype_; empty.nbytes = 0;
            out_grads.push_back(Storage{std::move(empty)});
        }
        return out_grads;
    }
    std::vector<Storage> out_grads;
    out_grads.push_back(Storage{std::move(dx1_cpu)});
    out_grads.push_back(Storage{std::move(dx2_cpu)});
    out_grads.push_back(Storage{std::move(dW_cpu)});
    if (has_bias) out_grads.push_back(Storage{std::move(db_cpu)});
    else {
        CpuStorage empty;
        empty.dtype = dtype_; empty.nbytes = 0;
        out_grads.push_back(Storage{std::move(empty)});
    }
    return out_grads;
}

TensorImplPtr bilinear_layer_op(const TensorImplPtr& x1,
                                  const TensorImplPtr& x2,
                                  const TensorImplPtr& weight,
                                  const TensorImplPtr& bias) {
    return BilinearLayerBackward::forward(x1, x2, weight, bias);
}
LUCID_REGISTER_OP(BilinearLayerBackward)

}  // namespace lucid
