// lucid/_C/ops/diffeq/RkErrorNorm.cpp
//
// Implements the fused embedded-error norm used by adaptive Runge-Kutta step
// control.  There is no backward node here on purpose: the result decides
// whether a step is accepted, which is control flow rather than a value the
// caller differentiates.  Returning a host double instead of a tensor makes
// that structural rather than a convention.
//
// Two paths, matching the engine's device split: a single-pass scalar loop on
// the CPU stream, and an MLX expression on the GPU stream whose evaluation is
// forced exactly once by the download at the end.

#include "RkErrorNorm.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/OpSchema.h"
#include "../../core/Scope.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../utils/Contiguous.h"

namespace lucid {

namespace {

// Schema carrier.  The op has no backward node to hang ``schema_v1`` on, but
// every op is expected to be registered, so a tag struct owns it.
struct RkErrorNormOp {
    static const OpSchema schema_v1;
};

const OpSchema RkErrorNormOp::schema_v1{"rk_error_norm",  1, AmpPolicy::KeepInput,
                                        /*det=*/true,
                                        /*note=*/"",
                                        /*in_arity=*/-1,
                                        /*out_arity=*/1,
                                        /*stable_ins=*/{}};

LUCID_REGISTER_OP(RkErrorNormOp)

// Single-pass CPU kernel.  ``j`` is the inner loop so the error estimate for
// element ``i`` is formed in a register and never stored — the stage buffers
// are read as a handful of parallel sequential streams, which prefetches
// well and keeps the whole thing to one pass over the data.
template <typename T>
double error_norm_cpu(const T* y0,
                      const T* y1,
                      const std::vector<const T*>& ks,
                      const std::vector<double>& scales,
                      std::size_t n,
                      double rtol,
                      double atol) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double err = 0.0;
        for (std::size_t j = 0; j < ks.size(); ++j)
            err += scales[j] * static_cast<double>(ks[j][i]);

        const double a0 = std::abs(static_cast<double>(y0[i]));
        const double a1 = std::abs(static_cast<double>(y1[i]));
        const double ratio = err / (atol + rtol * (a0 > a1 ? a0 : a1));
        acc += ratio * ratio;
    }
    return std::sqrt(acc / static_cast<double>(n));
}

// Reads element 0 of a freshly downloaded one-element CPU buffer.
double read_scalar(const CpuStorage& cs, Dtype dtype) {
    if (dtype == Dtype::F64)
        return *reinterpret_cast<const double*>(cs.ptr.get());
    return static_cast<double>(*reinterpret_cast<const float*>(cs.ptr.get()));
}

}  // namespace

// Validate the operands against ``y0``, materialise any views, then dispatch
// to the device-appropriate kernel.  Terms whose folded scale is exactly zero
// contribute nothing and are skipped; when none survive the error estimate is
// identically zero, so the ratio is zero and the step is always accepted.
double rk_error_norm_op(const TensorImplPtr& y0,
                        const TensorImplPtr& y1,
                        const std::vector<TensorImplPtr>& ks,
                        const std::vector<double>& coeffs,
                        double dt,
                        double rtol,
                        double atol) {
    Validator::input(y0, "rk_error_norm.y0").non_null();
    Validator::input(y1, "rk_error_norm.y1").non_null();
    if (ks.size() != coeffs.size())
        ErrorBuilder("rk_error_norm").fail("ks and coeffs must have the same length");

    const Dtype dtype = y0->dtype();
    const Device device = y0->device();
    const Shape shape = y0->shape();

    if (dtype != Dtype::F32 && dtype != Dtype::F64)
        throw NotImplementedError("rk_error_norm: dtype " + std::string(dtype_name(dtype)) +
                                  " is not supported; promote to float32 or float64 first");

    auto check = [&](const TensorImplPtr& t, const char* label) {
        Validator::input(t, label).non_null();
        if (t->dtype() != dtype)
            throw DtypeMismatch(std::string(dtype_name(dtype)), std::string(dtype_name(t->dtype())),
                                "rk_error_norm");
        if (t->device() != device)
            throw DeviceMismatch(std::string(device_name(device)),
                                 std::string(device_name(t->device())), "rk_error_norm");
        if (t->shape() != shape)
            throw ShapeMismatch(shape, t->shape(), "rk_error_norm");
    };
    check(y1, "rk_error_norm.y1");
    for (const auto& k : ks)
        check(k, "rk_error_norm.ks");

    OpScopeFull scope{"rk_error_norm", device, dtype, shape};

    std::vector<double> scales;
    std::vector<std::size_t> live;
    scales.reserve(ks.size());
    live.reserve(ks.size());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const double scale = dt * coeffs[i];
        if (scale == 0.0)
            continue;
        scales.push_back(scale);
        live.push_back(i);
    }
    if (live.empty())
        return 0.0;

    const std::size_t n = shape_numel(shape);
    if (n == 0)
        return 0.0;

    // Views have to be materialised for both paths.  No gradient is wired:
    // this value is control flow, so attaching a graph here would build nodes
    // that nothing can ever consume.
    NoGradGuard no_grad;
    const TensorImplPtr y0_c = y0->is_contiguous() ? y0 : contiguous_op(y0);
    const TensorImplPtr y1_c = y1->is_contiguous() ? y1 : contiguous_op(y1);
    std::vector<TensorImplPtr> ks_c;
    ks_c.reserve(live.size());
    for (const std::size_t idx : live) {
        const TensorImplPtr& k = ks[idx];
        ks_c.push_back(k->is_contiguous() ? k : contiguous_op(k));
    }

    if (device == Device::CPU) {
        const auto& c0 = std::get<CpuStorage>(y0_c->storage());
        const auto& c1 = std::get<CpuStorage>(y1_c->storage());
        if (dtype == Dtype::F64) {
            std::vector<const double*> kp;
            kp.reserve(ks_c.size());
            for (const auto& k : ks_c)
                kp.push_back(
                    reinterpret_cast<const double*>(std::get<CpuStorage>(k->storage()).ptr.get()));
            return error_norm_cpu<double>(reinterpret_cast<const double*>(c0.ptr.get()),
                                          reinterpret_cast<const double*>(c1.ptr.get()), kp, scales,
                                          n, rtol, atol);
        }
        std::vector<const float*> kp;
        kp.reserve(ks_c.size());
        for (const auto& k : ks_c)
            kp.push_back(
                reinterpret_cast<const float*>(std::get<CpuStorage>(k->storage()).ptr.get()));
        return error_norm_cpu<float>(reinterpret_cast<const float*>(c0.ptr.get()),
                                     reinterpret_cast<const float*>(c1.ptr.get()), kp, scales, n,
                                     rtol, atol);
    }

    // GPU: build the whole reduction as one lazy MLX expression so the
    // download below is the single point where it evaluates.
    namespace mx = ::mlx::core;
    const mx::array& a0 = *std::get<GpuStorage>(y0_c->storage()).arr;
    const mx::array& a1 = *std::get<GpuStorage>(y1_c->storage()).arr;
    const mx::Dtype mdt = a0.dtype();

    mx::array err = mx::multiply(*std::get<GpuStorage>(ks_c[0]->storage()).arr,
                                 gpu::mlx_scalar(scales[0], mdt));
    for (std::size_t j = 1; j < ks_c.size(); ++j)
        err = mx::add(err, mx::multiply(*std::get<GpuStorage>(ks_c[j]->storage()).arr,
                                        gpu::mlx_scalar(scales[j], mdt)));

    mx::array tol =
        mx::add(gpu::mlx_scalar(atol, mdt),
                mx::multiply(gpu::mlx_scalar(rtol, mdt), mx::maximum(mx::abs(a0), mx::abs(a1))));
    mx::array ratio = mx::divide(err, tol);
    mx::array out = mx::sqrt(mx::mean(mx::multiply(ratio, ratio), /*keepdims=*/false));

    GpuStorage gs = gpu::wrap_mlx_array(std::move(out), dtype);
    const CpuStorage host = gpu::download_gpu_to_cpu(gs, Shape{});
    return read_scalar(host, dtype);
}

}  // namespace lucid
