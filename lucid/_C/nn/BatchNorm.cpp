// lucid/_C/nn/BatchNorm.cpp
//
// Training-mode Batch Normalization for 1-D, 2-D, and 3-D inputs.
//
// For each channel c, statistics are computed over the (B, S_total) axes:
//   mean_c  = mean over all (b, spatial) pairs
//   rstd_c  = 1 / sqrt(var_c + eps)
//   y_{b,c,...} = (x_{b,c,...} - mean_c) * rstd_c * gamma_c + beta_c
//
// The forward delegates to IBackend::batch_norm_forward, which returns
// [y, mean, rstd].  The backward delegates to IBackend::batch_norm_backward,
// which returns [dx, d_gamma, d_beta].  Running-stats inference is handled by
// BatchNormEvalBackward in NormExt.h.

#include "BatchNorm.h"

#include <memory>
#include <vector>

#include <mlx/array.h>
#include <mlx/transforms.h>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../backend/IBackend.h"
#include "../core/AmpPolicy.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "../kernel/BinaryKernel.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/ufunc/Astype.h"

namespace lucid {

template <>
const OpSchema BatchNorm1dBackward::schema_v1{"batch_norm1d", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm2dBackward::schema_v1{"batch_norm", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm3dBackward::schema_v1{"batch_norm3d", 1, AmpPolicy::ForceFP32, true};

template <int N>
TensorImplPtr BatchNormNdBackward<N>::forward(const TensorImplPtr& x,
                                              const TensorImplPtr& gamma,
                                              const TensorImplPtr& beta,
                                              double eps,
                                              const TensorImplPtr& running_mean,
                                              const TensorImplPtr& running_var,
                                              double momentum) {
    if (!x || !gamma || !beta)
        ErrorBuilder("batch_norm").fail("null input");
    if (x->device() != gamma->device() || x->device() != beta->device())
        throw DeviceMismatch(std::string(device_name(x->device())),
                             std::string(device_name(gamma->device())), "batch_norm");
    // Rank check: x must be (B, C, S_0, ..., S_{N-1}).
    if (x->device() == Device::CPU &&
        (!x->is_contiguous() || !gamma->is_contiguous() || !beta->is_contiguous()))
        if (static_cast<int>(x->shape().size()) != N + 2)
            throw ShapeMismatch(x->shape(), Shape{}, "batch_norm: x rank mismatch");
    if (gamma->shape().size() != 1 || beta->shape().size() != 1)
        throw ShapeMismatch(gamma->shape(), beta->shape(), "batch_norm: γ, β must be 1-D");

    // 3.3 AMP plumbing: schema_v1.amp_policy == ForceFP32.  Under
    // ``AutocastGuard(F16)`` SchemaGuard returns ``Dtype::F32`` regardless
    // of input dtype — BN's batch statistics are numerically sensitive
    // and running them in F16 risks catastrophic cancellation of the
    // mean/variance reductions.  The cast MUST happen before the strict
    // ``x->dtype() != gamma->dtype()`` check below: under autocast, the
    // surrounding Conv has cast x to F16 while gamma / beta on the
    // ``nn.BatchNorm`` Parameter slots are still F32.  After the cast
    // all three operands share ``eff_dt`` and the dtype-match invariant
    // holds.  Outside an autocast scope this is a no-op (``astype_op``
    // returns the input unchanged when dtypes already match).
    //
    // ``astype_op`` (not ``maybe_cast_for_kernel``) is used so the cast
    // tensors carry an ``AstypeBackward`` grad_fn.  Without this the
    // F32-cast x_eff has requires_grad=false and ``wire_autograd`` would
    // drop the entire BN backward chain under AMP.
    SchemaGuard sg{BatchNormNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    // Trivially true after the AMP cast above — kept as a defensive
    // assertion in case ``astype_op`` ever returns an input whose dtype
    // doesn't match eff_dt (would indicate a backend bug).
    if (x_eff->dtype() != gamma_eff->dtype() || x_eff->dtype() != beta_eff->dtype())
        throw DtypeMismatch(std::string(dtype_name(x_eff->dtype())),
                            std::string(dtype_name(gamma_eff->dtype())), "batch_norm");

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    int S[N > 0 ? N : 1];
    int spatial_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        spatial_total *= S[i];
    }
    if (gamma_eff->shape()[0] != C || beta_eff->shape()[0] != C)
        throw ShapeMismatch(gamma_eff->shape(), x_eff->shape(),
                            "batch_norm: γ/β must have length C");

    OpScopeFull scope{BatchNormNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt,
                      x_eff->shape()};

    // batch_norm_forward returns [y, mean, rstd].
    auto forward = backend::Dispatcher::for_device(x_eff->device())
                       .batch_norm_forward(x_eff->storage(), gamma_eff->storage(),
                                           beta_eff->storage(), B, C, spatial_total, N, eps,
                                           x_eff->shape(), eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);
    auto bwd = std::make_shared<BatchNormNdBackward<N>>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    // 3.4+ Phase A.4: forward slot 3 holds xnorm = (x - mean) * rstd (MLX
    // path only — MPSGraph dispatch + CPU return an empty Storage here and
    // the backend backward then recomputes inline).  Saving the lazy MLX
    // intermediate is zero forward cost — it's already in the graph for
    // ``y = xnorm * γ + β`` — and lets backward skip the 2 element-wise
    // ops (subtract + multiply) that recompute it.
    if (forward.size() >= 4) {
        bwd->saved_xnorm_ = std::move(forward[3]);
    }
    bwd->B_ = B;
    bwd->C_ = C;
    for (int i = 0; i < N; ++i)
        bwd->S_[i] = S[i];
    bwd->eps_ = eps;

    // Running-stats EMA update — reuses the SAME mean+rstd the backend
    // just computed (saved on bwd) so we avoid a second mean+var
    // reduction over x.  Profile (Mac Studio M4 Max, ResNet-18 BS=32):
    // the prior separate ``_update_running_stats`` recomputed mean+var
    // for each of 16 BNs (~1.5 ms / step of duplicated GPU work);
    // folding it in here drops that to zero.  Variance is recovered
    // from rstd via ``var = 1/rstd² - eps`` — a few cheap elementwise
    // ops on a (C,) tensor.
    if (running_mean) {
        int n_total = B;
        for (int i = 0; i < N; ++i) n_total *= S[i];
        const double m = momentum;
        const double unbiased_factor =
            (n_total > 1) ? double(n_total) / double(n_total - 1) : 1.0;

        const Dtype buf_dt = running_mean->dtype();
        backend::IBackend& be = backend::Dispatcher::for_device(x_eff->device());

        // saved_mean/rstd are at shape (1, C, 1, ..., 1); reshape to (C,)
        // for the elementwise update.  Backend reshape is metadata-only.
        Shape kept_shape;
        kept_shape.reserve(2 + N);
        kept_shape.push_back(1);
        kept_shape.push_back(C);
        for (int i = 0; i < N; ++i) kept_shape.push_back(1);
        Shape stat_shape{static_cast<std::int64_t>(C)};

        Storage saved_mean_C = be.reshape(bwd->saved_mean_, kept_shape, stat_shape, eff_dt);
        Storage saved_rstd_C = be.reshape(bwd->saved_rstd_, kept_shape, stat_shape, eff_dt);

        // var = 1 / rstd^2 - eps
        Storage rstd_sq    = be.square(saved_rstd_C, stat_shape, buf_dt);
        Storage inv_rstd_sq = be.reciprocal(rstd_sq, stat_shape, buf_dt);
        Storage var        = be.add_scalar(inv_rstd_sq, stat_shape, buf_dt, -eps);

        // new_rm = (1-m) * running_mean + m * mean
        Storage rm_scaled  = be.mul_scalar(running_mean->storage(), stat_shape, buf_dt, 1.0 - m);
        Storage bm_scaled  = be.mul_scalar(saved_mean_C, stat_shape, buf_dt, m);
        Storage new_rm     = be.add(rm_scaled, bm_scaled, stat_shape, buf_dt);

        // new_rv = (1-m) * running_var + (m * n/(n-1)) * var
        Storage rv_scaled  = be.mul_scalar(running_var->storage(), stat_shape, buf_dt, 1.0 - m);
        Storage bv_scaled  = be.mul_scalar(var, stat_shape, buf_dt, m * unbiased_factor);
        Storage new_rv     = be.add(rv_scaled, bv_scaled, stat_shape, buf_dt);

        running_mean->mutable_storage() = std::move(new_rm);
        running_var->mutable_storage()  = std::move(new_rv);

        if (x_eff->device() == Device::GPU) {
            std::vector<mlx::core::array> arrs;
            arrs.reserve(2);
            if (const auto* gs = std::get_if<GpuStorage>(&running_mean->storage())) {
                if (gs->arr) arrs.push_back(*gs->arr);
            }
            if (const auto* gs = std::get_if<GpuStorage>(&running_var->storage())) {
                if (gs->arr) arrs.push_back(*gs->arr);
            }
            if (!arrs.empty()) mlx::core::async_eval(arrs);
        }
    }

    // saved_inputs_[0..2] hold {x, gamma, beta} at eff_dt.
    kernel::NaryKernel<BatchNormNdBackward<N>, 3>::wire_autograd(
        std::move(bwd), {x_eff, gamma_eff, beta_eff}, out);
    return out;
}

template <int N>
std::vector<Storage> BatchNormNdBackward<N>::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int i = 0; i < N; ++i)
        spatial_total *= this->S_[i];

    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(this->device_)
        .batch_norm_backward(this->saved_inputs_[0], this->saved_inputs_[1], this->saved_mean_,
                             this->saved_rstd_, this->saved_xnorm_, grad_out, this->B_, this->C_,
                             spatial_total, N, this->input_shapes_[0], this->dtype_, this->eps_);
}

template class BatchNormNdBackward<1>;
template class BatchNormNdBackward<2>;
template class BatchNormNdBackward<3>;

TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                              const TensorImplPtr& gamma,
                              const TensorImplPtr& beta,
                              double eps,
                              const TensorImplPtr& running_mean,
                              const TensorImplPtr& running_var,
                              double momentum) {
    return BatchNorm1dBackward::forward(x, gamma, beta, eps, running_mean, running_var, momentum);
}
TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            double eps,
                            const TensorImplPtr& running_mean,
                            const TensorImplPtr& running_var,
                            double momentum) {
    return BatchNorm2dBackward::forward(x, gamma, beta, eps, running_mean, running_var, momentum);
}
TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                              const TensorImplPtr& gamma,
                              const TensorImplPtr& beta,
                              double eps,
                              const TensorImplPtr& running_mean,
                              const TensorImplPtr& running_var,
                              double momentum) {
    return BatchNorm3dBackward::forward(x, gamma, beta, eps, running_mean, running_var, momentum);
}

LUCID_REGISTER_OP(BatchNorm1dBackward)
LUCID_REGISTER_OP(BatchNorm2dBackward)
LUCID_REGISTER_OP(BatchNorm3dBackward)

// ── Running-stats update (BatchNorm + InstanceNorm hot path) ──────────────
//
// Collapses the Python composition
//
//     batch_mean = x.mean(reduce_dims)
//     batch_var  = x.var(reduce_dims, correction=0)
//     new_rm = (1-eff)*running_mean + eff*batch_mean
//     new_rv = (1-eff)*running_var  + (eff*n/(n-1))*batch_var
//     buffers["running_mean"] = new_rm
//     buffers["running_var"]  = new_rv
//     buffers["num_batches_tracked"] += 1
//     eval_tensors_async([running_mean, running_var, num_batches_tracked])
//
// into a single C++ entry.  Profile (Mac Studio M4 Max, ResNet-18 BS=32) shows
// the Python composition takes ~8.8 ms / forward via ~160 pybind11 crossings
// (16 BN × ~10 Python ops); a single C++ call eliminates the per-arithmetic-op
// crossing cost.  Behaviour matches Lucid's existing Python `_update_running_stats`
// 1:1, including the AutocastGuard(F32) override when AMP is active.
//
// Constraints:
//   - momentum must be a finite double (cumulative mode — Python's
//     momentum=None — is rejected so callers can keep that on the Python
//     path; cumulative requires reading num_batches_tracked.item() which
//     forces a GPU sync, defeating the optimisation).
//   - running_mean / running_var must share the same buffer dtype and the
//     shape (C,) where C == x->shape()[1].
//   - reduce_axes is the list of dims to reduce over: e.g. [0, 2, 3] for
//     BatchNorm2d, [2, 3] for InstanceNorm2d.
void batch_norm_update_running_stats(std::shared_ptr<TensorImpl> running_mean,
                                     std::shared_ptr<TensorImpl> running_var,
                                     const std::shared_ptr<TensorImpl>& x,
                                     std::vector<int> reduce_axes,
                                     double momentum,
                                     bool unbiased_var) {
    if (!x || !running_mean || !running_var) {
        ErrorBuilder("batch_norm_update_running_stats").fail("null tensor input");
    }

    if (!std::isfinite(momentum)) {
        ErrorBuilder("batch_norm_update_running_stats")
            .fail("momentum must be a finite double; cumulative mode (None) "
                  "is not supported in the C++ fast path");
    }

    Device device = x->device();
    backend::IBackend& be = backend::Dispatcher::for_device(device);
    const Shape& x_shape = x->shape();

    // Product of reduced-dim sizes — needed for the (n/(n-1)) Bessel factor.
    int n_total = 1;
    for (int d : reduce_axes) {
        if (d < 0 || d >= static_cast<int>(x_shape.size())) {
            ErrorBuilder("batch_norm_update_running_stats")
                .fail("reduce axis out of range");
        }
        n_total *= static_cast<int>(x_shape[d]);
    }

    Dtype buf_dt = running_mean->dtype();

    // AMP override — matches the Python AutocastGuard(F32) the previous
    // implementation engaged for the duration of the update.  Running buffers
    // are always kept at F32 regardless of the outer autocast scope.
    std::unique_ptr<amp::AutocastGuard> amp_off;
    if (amp::is_active()) {
        amp_off = std::make_unique<amp::AutocastGuard>(Dtype::F32);
    }

    // Cast x → buf_dt if needed.  ``astype`` is a no-op when dtypes match.
    Storage x_eff = x->storage();
    Dtype x_dt = x->dtype();
    if (x_dt != buf_dt) {
        x_eff = be.astype(x_eff, x_shape, x_dt, buf_dt);
    }

    backend::ReduceOpts opts;
    opts.axes = reduce_axes;
    opts.keepdims = false;
    Shape stat_shape{static_cast<std::int64_t>(x_shape[1])};

    Storage batch_mean = be.reduce_mean(x_eff, x_shape, opts, buf_dt);
    Storage batch_var  = be.variance(x_eff, x_shape, opts, buf_dt);

    const double eff = momentum;
    const double unbiased_factor =
        (unbiased_var && n_total > 1) ? double(n_total) / double(n_total - 1) : 1.0;

    // new_rm = (1-eff) * running_mean + eff * batch_mean
    Storage rm_scaled = be.mul_scalar(running_mean->storage(),
                                      running_mean->shape(), buf_dt, 1.0 - eff);
    Storage bm_scaled = be.mul_scalar(batch_mean, stat_shape, buf_dt, eff);
    Storage new_rm    = be.add(rm_scaled, bm_scaled, stat_shape, buf_dt);

    // new_rv = (1-eff) * running_var + (eff * unbiased_factor) * batch_var
    Storage rv_scaled = be.mul_scalar(running_var->storage(),
                                      running_var->shape(), buf_dt, 1.0 - eff);
    Storage bv_scaled = be.mul_scalar(batch_var, stat_shape, buf_dt, eff * unbiased_factor);
    Storage new_rv    = be.add(rv_scaled, bv_scaled, stat_shape, buf_dt);

    // In-place storage replacement.  Matches the Python pattern
    // ``self._buffers["running_mean"] = new_rm`` — the TensorImpl identity
    // is preserved; only its underlying storage is swapped.
    running_mean->mutable_storage() = std::move(new_rm);
    running_var->mutable_storage()  = std::move(new_rv);

    // GPU: break the MLX lazy expression chain so subsequent training steps
    // don't carry an ever-growing parent graph through running_mean's lineage.
    // ``async_eval`` schedules without blocking the CPU.  num_batches_tracked
    // is left to the Python caller (its dtype is I64 which not all backends
    // support via add_scalar; the increment is one cheap op in Python).
    if (device == Device::GPU) {
        std::vector<mlx::core::array> arrs;
        arrs.reserve(2);
        if (const auto* gs = std::get_if<GpuStorage>(&running_mean->storage())) {
            if (gs->arr) arrs.push_back(*gs->arr);
        }
        if (const auto* gs = std::get_if<GpuStorage>(&running_var->storage())) {
            if (gs->arr) arrs.push_back(*gs->arr);
        }
        if (!arrs.empty()) mlx::core::async_eval(arrs);
    }
}

}  // namespace lucid
