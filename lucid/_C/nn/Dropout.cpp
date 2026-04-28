#include "Dropout.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema DropoutBackward::schema_v1{
    "dropout", 1, AmpPolicy::KeepInput,
    /*deterministic=*/false,
    "uses Generator-driven Bernoulli mask; reproducible only with explicit seed"};

TensorImplPtr DropoutBackward::forward(const TensorImplPtr& a,
                                       double p,
                                       bool training,
                                       Generator* gen) {
    Validator::input(a, "dropout.a").non_null();
    if (a->device_ == Device::CPU && !a->is_contiguous())
        ErrorBuilder("dropout").not_implemented(
            "non-contiguous input not supported (call .contiguous() first)");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropout").fail("p must be in [0, 1)");

    OpScopeFull scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    const std::size_t numel = a->numel();

    // Inference mode (or zero drop) → pure pass-through clone (so engine
    // owns its own buffer; no graph wiring needed if !requires_grad).
    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage_, numel, a->dtype_, a->device_);
        auto out =
            std::make_shared<TensorImpl>(std::move(clone), a->shape_, a->dtype_, a->device_, false);
        if (!GradMode::is_enabled() || !a->requires_grad_)
            return out;

        auto a_edge = detail::ensure_grad_fn(a);
        auto bwd = std::make_shared<DropoutBackward>();
        bwd->input_shapes_ = {a->shape_};
        bwd->out_shape_ = a->shape_;
        bwd->dtype_ = a->dtype_;
        bwd->device_ = a->device_;
        bwd->input_tensors_ = {a};
        bwd->p_ = 0.0;  // identity backward
        // mask_ is empty/uninitialized — apply() detects and short-circuits.
        bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
        bwd->set_saved_versions({a->version_});

        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    // Training: sample Bernoulli(1-p) mask, y = x * mask / (1-p).
    Generator& g = gen ? *gen : default_generator();
    Storage mask = bernoulli_mask_storage_shape(1.0 - p, a->shape_, a->dtype_, a->device_, g);
    const double scale = 1.0 / (1.0 - p);

    Storage scaled_mask = mul_scalar_storage(mask, scale, numel, a->dtype_, a->device_);
    Storage y = multiply_storages(a->storage_, scaled_mask, numel, a->dtype_, a->device_);

    auto out = std::make_shared<TensorImpl>(std::move(y), a->shape_, a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<DropoutBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = a->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->p_ = p;
    bwd->mask_ = std::move(scaled_mask);  // already scaled by 1/(1-p)
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> DropoutBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    // Inference / p=0 path: identity backward (mask_ is empty / default-constructed).
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return DropoutBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(DropoutBackward)

// ===================================================================
// Helpers shared by the dropout family
// ===================================================================

namespace {

// Build a (B, C, 1, ..., 1) shape from an input shape.
Shape channel_mask_shape(const Shape& in) {
    if (in.size() < 2) {
        ErrorBuilder("dropoutnd").fail("expected ndim >= 2, got" + std::to_string(in.size()));
    }
    Shape m(in.size(), 1);
    m[0] = in[0];
    m[1] = in[1];
    return m;
}

// (B, 1, 1, ..., 1) shape from input.
Shape sample_mask_shape(const Shape& in) {
    if (in.empty())
        ErrorBuilder("drop_path").fail("input must have ≥1 dim");
    Shape m(in.size(), 1);
    m[0] = in[0];
    return m;
}

// Tile a (B, C, 1, 1, ...) mask up to the full input shape via repeated
// element-wise multiplication using broadcast_back semantics. Since the
// engine's `multiply_storages` is element-wise (no broadcast), we manually
// expand the mask buffer to the full shape.
template <typename T>
void expand_channel_mask_typed(
    const T* mask, T* out, std::size_t B, std::size_t C, std::size_t spatial) {
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t c = 0; c < C; ++c) {
            const T v = mask[b * C + c];
            T* dst = out + (b * C + c) * spatial;
            for (std::size_t s = 0; s < spatial; ++s)
                dst[s] = v;
        }
    }
}

CpuStorage expand_channel_mask(const CpuStorage& mask, const Shape& full, Dtype dt) {
    const std::size_t B = static_cast<std::size_t>(full[0]);
    const std::size_t C = static_cast<std::size_t>(full[1]);
    std::size_t spatial = 1;
    for (std::size_t i = 2; i < full.size(); ++i) {
        spatial *= static_cast<std::size_t>(full[i]);
    }
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = B * C * spatial * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dt) {
        case Dtype::F32:
            expand_channel_mask_typed<float>(reinterpret_cast<const float*>(mask.ptr.get()),
                                             reinterpret_cast<float*>(out.ptr.get()), B, C,
                                             spatial);
            break;
        case Dtype::F64:
            expand_channel_mask_typed<double>(reinterpret_cast<const double*>(mask.ptr.get()),
                                              reinterpret_cast<double*>(out.ptr.get()), B, C,
                                              spatial);
            break;
        default:
            ErrorBuilder("dropoutnd").not_implemented("dtype not supported");
    }
    return out;
}

template <typename T>
void expand_sample_mask_typed(const T* mask, T* out, std::size_t B, std::size_t per_sample) {
    for (std::size_t b = 0; b < B; ++b) {
        const T v = mask[b];
        T* dst = out + b * per_sample;
        for (std::size_t s = 0; s < per_sample; ++s)
            dst[s] = v;
    }
}

CpuStorage expand_sample_mask(const CpuStorage& mask, const Shape& full, Dtype dt) {
    const std::size_t B = static_cast<std::size_t>(full[0]);
    std::size_t per = 1;
    for (std::size_t i = 1; i < full.size(); ++i) {
        per *= static_cast<std::size_t>(full[i]);
    }
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = B * per * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    switch (dt) {
        case Dtype::F32:
            expand_sample_mask_typed<float>(reinterpret_cast<const float*>(mask.ptr.get()),
                                            reinterpret_cast<float*>(out.ptr.get()), B, per);
            break;
        case Dtype::F64:
            expand_sample_mask_typed<double>(reinterpret_cast<const double*>(mask.ptr.get()),
                                             reinterpret_cast<double*>(out.ptr.get()), B, per);
            break;
        default:
            ErrorBuilder("drop_path").not_implemented("dtype not supported");
    }
    return out;
}

}  // namespace

// ===================================================================
// DropoutNd (channel-wise: dropout1d/2d/3d)
// ===================================================================

const OpSchema DropoutNdBackward::schema_v1{
    "dropoutnd", 1, AmpPolicy::KeepInput, /*deterministic=*/false,
    "Bernoulli channel mask; reproducible only with explicit seed"};

TensorImplPtr DropoutNdBackward::forward(const TensorImplPtr& a,
                                         double p,
                                         bool training,
                                         Generator* gen) {
    Validator::input(a, "dropoutnd.a").non_null();
    if (a->device_ == Device::CPU && !a->is_contiguous())
        ErrorBuilder("dropoutnd")
            .not_implemented("non-contiguous input not supported (call .contiguous())");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropoutnd").fail("p must be in [0, 1)");

    OpScopeFull scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage_, numel, a->dtype_, a->device_);
        auto out =
            std::make_shared<TensorImpl>(std::move(clone), a->shape_, a->dtype_, a->device_, false);
        if (!GradMode::is_enabled() || !a->requires_grad_)
            return out;
        auto a_edge = detail::ensure_grad_fn(a);
        auto bwd = std::make_shared<DropoutNdBackward>();
        bwd->input_shapes_ = {a->shape_};
        bwd->out_shape_ = a->shape_;
        bwd->dtype_ = a->dtype_;
        bwd->device_ = a->device_;
        bwd->input_tensors_ = {a};
        bwd->p_ = 0.0;
        bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
        bwd->set_saved_versions({a->version_});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    // Sample a (B, C, 1, 1, ...) Bernoulli mask (broadcast-shape).
    Shape mask_shape = channel_mask_shape(a->shape_);
    const std::size_t mask_numel =
        static_cast<std::size_t>(mask_shape[0]) * static_cast<std::size_t>(mask_shape[1]);
    Generator& g = gen ? *gen : default_generator();
    Storage small_mask =
        bernoulli_mask_storage_shape(1.0 - p, mask_shape, a->dtype_, a->device_, g);
    const double scale = 1.0 / (1.0 - p);
    Storage scaled_small = mul_scalar_storage(small_mask, scale, mask_numel, a->dtype_, a->device_);
    Storage exp_storage;
    Storage y;
    if (a->device_ == Device::GPU) {
        // MLX broadcasts on multiply; expand the (B,C,1,...) mask to full shape so the
        // saved mask exactly matches grad_out shape during backward.
        const auto& gm = std::get<GpuStorage>(scaled_small);
        auto full_mask = ::mlx::core::broadcast_to(*gm.arr, gpu::to_mlx_shape(a->shape_));
        const auto& gx = std::get<GpuStorage>(a->storage_);
        auto out = ::mlx::core::multiply(*gx.arr, full_mask);
        exp_storage = Storage{gpu::wrap_mlx_array(std::move(full_mask), a->dtype_)};
        y = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        // CPU: explicit expand to full shape.
        CpuStorage expanded =
            expand_channel_mask(std::get<CpuStorage>(scaled_small), a->shape_, a->dtype_);
        exp_storage = Storage{std::move(expanded)};
        y = multiply_storages(a->storage_, exp_storage, numel, a->dtype_, a->device_);
    }
    auto out = std::make_shared<TensorImpl>(std::move(y), a->shape_, a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;
    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<DropoutNdBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = a->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->p_ = p;
    bwd->mask_ = std::move(exp_storage);  // full-shape, scaled
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
    bwd->set_saved_versions({a->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> DropoutNdBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr dropoutnd_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return DropoutNdBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(DropoutNdBackward)

// ===================================================================
// AlphaDropout (SELU-friendly)
// ===================================================================
//
// Following Klambauer et al. (2017) and the canonical PyTorch
// implementation: keep_prob = 1-p, alpha' = -lambda * alpha. We build
//   y = a · (x · mask + alpha' · (1 - mask)) + b
// where  a = (keep_prob · (1 + p · alpha'²))^-0.5
//        b = -a · p · alpha'
// Backward: dx = a · mask · g  (the additive constant has no x dependence)

namespace {
constexpr double kSeluAlpha = 1.6732632423543772;
constexpr double kSeluLambda = 1.0507009873554805;
constexpr double kAlphaPrime = -kSeluLambda * kSeluAlpha;  // ≈ -1.7581
}  // namespace

const OpSchema AlphaDropoutBackward::schema_v1{
    "alpha_dropout", 1, AmpPolicy::KeepInput, /*deterministic=*/false,
    "Bernoulli mask + SELU rescaling; reproducible only with explicit seed"};

TensorImplPtr AlphaDropoutBackward::forward(const TensorImplPtr& a,
                                            double p,
                                            bool training,
                                            Generator* gen) {
    Validator::input(a, "alpha_dropout.a").non_null();
    if (a->device_ == Device::CPU && !a->is_contiguous())
        ErrorBuilder("alpha_dropout").not_implemented("non-contiguous input not supported");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("alpha_dropout").fail("p must be in [0, 1)");

    OpScopeFull scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage_, numel, a->dtype_, a->device_);
        auto out =
            std::make_shared<TensorImpl>(std::move(clone), a->shape_, a->dtype_, a->device_, false);
        if (!GradMode::is_enabled() || !a->requires_grad_)
            return out;
        auto a_edge = detail::ensure_grad_fn(a);
        auto bwd = std::make_shared<AlphaDropoutBackward>();
        bwd->input_shapes_ = {a->shape_};
        bwd->out_shape_ = a->shape_;
        bwd->dtype_ = a->dtype_;
        bwd->device_ = a->device_;
        bwd->input_tensors_ = {a};
        bwd->p_ = 0.0;
        bwd->a_coef_ = 1.0;
        bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
        bwd->set_saved_versions({a->version_});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    const double keep = 1.0 - p;
    const double a_coef = std::pow(keep * (1.0 + p * kAlphaPrime * kAlphaPrime), -0.5);
    const double b_coef = -a_coef * p * kAlphaPrime;

    Generator& g = gen ? *gen : default_generator();
    Storage mask = bernoulli_mask_storage_shape(keep, a->shape_, a->dtype_, a->device_, g);

    // x·mask
    Storage x_mask = multiply_storages(a->storage_, mask, numel, a->dtype_, a->device_);
    // (1 - mask)·alpha' = alpha' - alpha'·mask
    Storage alpha_mask = mul_scalar_storage(mask, kAlphaPrime, numel, a->dtype_, a->device_);
    Storage neg_alpha_mask = mul_scalar_storage(alpha_mask, -1.0, numel, a->dtype_, a->device_);
    Storage one_m_mask_ap =
        add_scalar_storage(neg_alpha_mask, kAlphaPrime, numel, a->dtype_, a->device_);
    // x·mask + (1 - mask)·alpha'
    Storage inner = add_storages(x_mask, one_m_mask_ap, numel, a->dtype_, a->device_);
    Storage scaled = mul_scalar_storage(inner, a_coef, numel, a->dtype_, a->device_);
    Storage y = add_scalar_storage(scaled, b_coef, numel, a->dtype_, a->device_);

    auto out = std::make_shared<TensorImpl>(std::move(y), a->shape_, a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;
    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<AlphaDropoutBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = a->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->p_ = p;
    bwd->a_coef_ = a_coef;
    bwd->mask_ = std::move(mask);
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
    bwd->set_saved_versions({a->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> AlphaDropoutBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    // dx = a_coef · mask · g
    Storage scaled_mask = mul_scalar_storage(mask_, a_coef_, numel, dtype_, device_);
    return {multiply_storages(grad_out, scaled_mask, numel, dtype_, device_)};
}

TensorImplPtr alpha_dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return AlphaDropoutBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(AlphaDropoutBackward)

// ===================================================================
// DropBlock (4-D spatial structured dropout)
// ===================================================================

const OpSchema DropBlockBackward::schema_v1{
    "drop_block", 1, AmpPolicy::KeepInput, /*deterministic=*/false,
    "Bernoulli + spatial dilation; reproducible only with explicit seed"};

namespace {

// Compute the block mask in CPU-typed memory. Inputs: per-position seed mask
// `m` of shape (B, C, H, W), block_size, output `block_mask` (same shape)
// is set to 1 wherever any seed in the block_size×block_size neighbourhood
// is 1 (i.e. dilation by max-pooling).
template <typename T>
void dilate_seed_mask_typed(const T* seed,
                            T* block_mask,
                            std::size_t B,
                            std::size_t C,
                            std::size_t H,
                            std::size_t W,
                            std::int64_t block_size) {
    const std::int64_t pad = block_size / 2;
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t c = 0; c < C; ++c) {
            const T* s = seed + (b * C + c) * H * W;
            T* m = block_mask + (b * C + c) * H * W;
            for (std::size_t y = 0; y < H; ++y) {
                for (std::size_t x = 0; x < W; ++x) {
                    T mx = T{0};
                    for (std::int64_t dy = 0; dy < block_size; ++dy) {
                        const std::int64_t yy = static_cast<std::int64_t>(y) + dy - pad;
                        if (yy < 0 || yy >= static_cast<std::int64_t>(H))
                            continue;
                        for (std::int64_t dx = 0; dx < block_size; ++dx) {
                            const std::int64_t xx = static_cast<std::int64_t>(x) + dx - pad;
                            if (xx < 0 || xx >= static_cast<std::int64_t>(W))
                                continue;
                            const T v = s[yy * W + xx];
                            if (v > mx)
                                mx = v;
                        }
                    }
                    m[y * W + x] = mx;
                }
            }
        }
    }
}

CpuStorage build_drop_block_mask(
    const CpuStorage& seed, const Shape& full, std::int64_t block_size, double p, Dtype dt) {
    const std::size_t B = static_cast<std::size_t>(full[0]);
    const std::size_t C = static_cast<std::size_t>(full[1]);
    const std::size_t H = static_cast<std::size_t>(full[2]);
    const std::size_t W = static_cast<std::size_t>(full[3]);
    const std::size_t total = B * C * H * W;
    CpuStorage block_mask;
    block_mask.dtype = dt;
    block_mask.nbytes = total * dtype_size(dt);
    block_mask.ptr = allocate_aligned_bytes(block_mask.nbytes);
    switch (dt) {
        case Dtype::F32:
            dilate_seed_mask_typed<float>(reinterpret_cast<const float*>(seed.ptr.get()),
                                          reinterpret_cast<float*>(block_mask.ptr.get()), B, C, H,
                                          W, block_size);
            break;
        case Dtype::F64:
            dilate_seed_mask_typed<double>(reinterpret_cast<const double*>(seed.ptr.get()),
                                           reinterpret_cast<double*>(block_mask.ptr.get()), B, C, H,
                                           W, block_size);
            break;
        default:
            ErrorBuilder("drop_block").not_implemented("dtype not supported");
    }
    // Convert to keep mask: keep = (1 - block_mask) / (1 - p)
    const double scale = 1.0 / (1.0 - p);
    switch (dt) {
        case Dtype::F32: {
            auto* p32 = reinterpret_cast<float*>(block_mask.ptr.get());
            const float fs = static_cast<float>(scale);
            for (std::size_t i = 0; i < total; ++i)
                p32[i] = (1.f - p32[i]) * fs;
            break;
        }
        case Dtype::F64: {
            auto* p64 = reinterpret_cast<double*>(block_mask.ptr.get());
            for (std::size_t i = 0; i < total; ++i)
                p64[i] = (1.0 - p64[i]) * scale;
            break;
        }
        default:
            break;
    }
    return block_mask;
}

}  // namespace

TensorImplPtr DropBlockBackward::forward(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen) {
    Validator::input(a, "drop_block.a").non_null();
    if (a->device_ == Device::CPU && !a->is_contiguous())
        ErrorBuilder("drop_block").not_implemented("non-contiguous input not supported");
    if (a->shape_.size() != 4)
        ErrorBuilder("drop_block").fail("input must be 4-D (N, C, H, W)");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("drop_block").fail("p must be in [0, 1)");
    if (block_size <= 0)
        ErrorBuilder("drop_block").fail("block_size must be > 0");

    OpScopeFull scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    const std::size_t numel = a->numel();
    const std::int64_t H = a->shape_[2];
    const std::int64_t W = a->shape_[3];

    const double feat = static_cast<double>(H * W);
    const double valid = static_cast<double>((H - block_size + 1) * (W - block_size + 1));
    const double gamma = p * feat / (block_size * block_size) / (valid + eps);

    Generator& g = gen ? *gen : default_generator();
    Storage keep_storage;
    if (a->device_ == Device::GPU) {
        // Native GPU path: pure mlx::core ops only.
        // 1) bernoulli seed on GPU
        // 2) pad seed by K/2 on H, W axes
        // 3) dilation = max over K*K shifted (B, C, H, W) slices
        // 4) keep = scale * (1 - dilation)
        Storage seed = bernoulli_mask_storage(gamma, numel, a->dtype_, Device::GPU, g);
        const auto& seed_g = std::get<GpuStorage>(seed);
        auto mlx_dt = gpu::to_mlx_dtype(a->dtype_);

        const int K = static_cast<int>(block_size);
        const int pad = K / 2;
        const int B = static_cast<int>(a->shape_[0]);
        const int C = static_cast<int>(a->shape_[1]);
        const int H = static_cast<int>(a->shape_[2]);
        const int W = static_cast<int>(a->shape_[3]);

        // bernoulli_mask_storage returns a flat (numel,) mlx array; reshape.
        auto seed_4d = ::mlx::core::reshape(*seed_g.arr, ::mlx::core::Shape{B, C, H, W});

        // Pad H and W axes by `pad` on both sides (zero fill).
        auto seed_pad = ::mlx::core::pad(
            seed_4d, std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {pad, pad}, {pad, pad}},
            ::mlx::core::array(0.0f, mlx_dt));

        // Dilation: max over all (dy, dx) in [0, K) × [0, K) shifted slices.
        ::mlx::core::array dilated = ::mlx::core::zeros(::mlx::core::Shape{B, C, H, W}, mlx_dt);
        for (int dy = 0; dy < K; ++dy) {
            for (int dx = 0; dx < K; ++dx) {
                auto s = ::mlx::core::slice(seed_pad, ::mlx::core::Shape{0, 0, dy, dx},
                                            ::mlx::core::Shape{B, C, dy + H, dx + W});
                if (dy == 0 && dx == 0)
                    dilated = s;
                else
                    dilated = ::mlx::core::maximum(dilated, s);
            }
        }

        auto one = ::mlx::core::array(1.0f, mlx_dt);
        auto scale = ::mlx::core::array(static_cast<float>(1.0 / (1.0 - p)), mlx_dt);
        auto keep = ::mlx::core::multiply(scale, ::mlx::core::subtract(one, dilated));
        keep_storage = Storage{gpu::wrap_mlx_array(std::move(keep), a->dtype_)};
    } else {
        Storage seed = bernoulli_mask_storage(gamma, numel, a->dtype_, Device::CPU, g);
        CpuStorage keep_mask =
            build_drop_block_mask(std::get<CpuStorage>(seed), a->shape_, block_size, p, a->dtype_);
        keep_storage = Storage{std::move(keep_mask)};
    }
    Storage y = multiply_storages(a->storage_, keep_storage, numel, a->dtype_, a->device_);

    auto out = std::make_shared<TensorImpl>(std::move(y), a->shape_, a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;
    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<DropBlockBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = a->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->p_ = p;
    bwd->mask_ = std::move(keep_storage);
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
    bwd->set_saved_versions({a->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> DropBlockBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr drop_block_op(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen) {
    return DropBlockBackward::forward(a, block_size, p, eps, gen);
}

LUCID_REGISTER_OP(DropBlockBackward)

// ===================================================================
// DropPath (stochastic depth)
// ===================================================================

const OpSchema DropPathBackward::schema_v1{
    "drop_path", 1, AmpPolicy::KeepInput, /*deterministic=*/false,
    "Per-sample Bernoulli; reproducible only with explicit seed"};

TensorImplPtr DropPathBackward::forward(const TensorImplPtr& a,
                                        double p,
                                        bool scale_by_keep,
                                        Generator* gen) {
    Validator::input(a, "drop_path.a").non_null();
    if (a->device_ == Device::CPU && !a->is_contiguous())
        ErrorBuilder("drop_path").not_implemented("non-contiguous input not supported");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("drop_path").fail("p must be in [0, 1)");

    OpScopeFull scope{schema_v1.name, a->device_, a->dtype_, a->shape_};
    const std::size_t numel = a->numel();

    if (p == 0.0) {
        Storage clone = clone_storage(a->storage_, numel, a->dtype_, a->device_);
        auto out =
            std::make_shared<TensorImpl>(std::move(clone), a->shape_, a->dtype_, a->device_, false);
        if (!GradMode::is_enabled() || !a->requires_grad_)
            return out;
        auto a_edge = detail::ensure_grad_fn(a);
        auto bwd = std::make_shared<DropPathBackward>();
        bwd->input_shapes_ = {a->shape_};
        bwd->out_shape_ = a->shape_;
        bwd->dtype_ = a->dtype_;
        bwd->device_ = a->device_;
        bwd->input_tensors_ = {a};
        bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
        bwd->set_saved_versions({a->version_});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }

    const double keep = 1.0 - p;
    const std::size_t B = static_cast<std::size_t>(a->shape_[0]);
    Shape sshape = sample_mask_shape(a->shape_);  // (B, 1, 1, ...)
    Generator& g = gen ? *gen : default_generator();
    Storage small = bernoulli_mask_storage_shape(keep, sshape, a->dtype_, a->device_, g);
    if (scale_by_keep && keep > 0.0) {
        small = mul_scalar_storage(small, 1.0 / keep, B, a->dtype_, a->device_);
    }
    Storage exp_storage;
    Storage y;
    if (a->device_ == Device::GPU) {
        const auto& gm = std::get<GpuStorage>(small);
        auto full_mask = ::mlx::core::broadcast_to(*gm.arr, gpu::to_mlx_shape(a->shape_));
        const auto& gx = std::get<GpuStorage>(a->storage_);
        auto out = ::mlx::core::multiply(*gx.arr, full_mask);
        exp_storage = Storage{gpu::wrap_mlx_array(std::move(full_mask), a->dtype_)};
        y = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        CpuStorage expanded = expand_sample_mask(std::get<CpuStorage>(small), a->shape_, a->dtype_);
        exp_storage = Storage{std::move(expanded)};
        y = multiply_storages(a->storage_, exp_storage, numel, a->dtype_, a->device_);
    }

    auto out = std::make_shared<TensorImpl>(std::move(y), a->shape_, a->dtype_, a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;
    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<DropPathBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = a->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->mask_ = std::move(exp_storage);
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
    bwd->set_saved_versions({a->version_});
    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> DropPathBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr drop_path_op(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen) {
    return DropPathBackward::forward(a, p, scale_by_keep, gen);
}

LUCID_REGISTER_OP(DropPathBackward)

}  // namespace lucid
