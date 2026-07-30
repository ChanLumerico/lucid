// lucid/_C/ops/diffeq/BroydenProbe.cpp
//
// Implements the fused Broyden iteration probe.  Two kernels, one per device
// family, behind the validation the public entry point performs once.
//
// The GPU path is the reason this op exists: it builds the reductions as one
// lazy MLX expression and downloads them together, so an iteration costs a
// single pipeline flush.  The CPU path has no flush to amortise but still
// wins, because the intermediate tensors the equivalent Python would allocate
// never come into being.

#include "BroydenProbe.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Scope.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../utils/Contiguous.h"
#include "Operand.h"

namespace lucid {

namespace {

// Sum of squares over ``n`` contiguous elements.
//
// Accumulated in ``double`` whatever the input precision: this feeds a
// convergence test, and letting a float32 accumulator lose the tail would
// make the test decide differently near the tolerance.
template <typename T>
double sum_squares(const T* data, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(data[i]);
        acc += v * v;
    }
    return acc;
}

// Read a 0-d storage of any supported dtype as a ``double``.
//
// ``info`` arrives as whatever the linear solve produced -- an integer status
// in practice -- so this covers more than the floating cases the reductions
// need.
double read_any_scalar(const CpuStorage& cs, Dtype dtype) {
    const void* p = cs.ptr.get();
    switch (dtype) {
    case Dtype::F64:
        return *reinterpret_cast<const double*>(p);
    case Dtype::F32:
        return static_cast<double>(*reinterpret_cast<const float*>(p));
    case Dtype::I64:
        return static_cast<double>(*reinterpret_cast<const std::int64_t*>(p));
    case Dtype::I32:
        return static_cast<double>(*reinterpret_cast<const std::int32_t*>(p));
    case Dtype::I16:
        return static_cast<double>(*reinterpret_cast<const std::int16_t*>(p));
    case Dtype::I8:
        return static_cast<double>(*reinterpret_cast<const std::int8_t*>(p));
    case Dtype::Bool:
        return *reinterpret_cast<const bool*>(p) ? 1.0 : 0.0;
    default:
        throw NotImplementedError("broyden_probe: info dtype " + std::string(dtype_name(dtype)) +
                                  " is not supported");
    }
}

}  // namespace

// Validate the operands, materialise any views, then dispatch to the
// device-appropriate kernel.  ``step`` is compared to ``residual`` by element
// count rather than shape: the linear solve hands back a column while the
// residual is flat, and reshaping one to match the other would allocate for
// nothing.
BroydenProbeResult broyden_probe_op(const TensorImplPtr& residual,
                                    const TensorImplPtr& step,
                                    const TensorImplPtr& state,
                                    const TensorImplPtr& info) {
    Validator::input(residual, "broyden_probe.residual").non_null();
    Validator::input(step, "broyden_probe.step").non_null();
    Validator::input(state, "broyden_probe.state").non_null();

    const Dtype dtype = residual->dtype();
    const Device device = residual->device();
    const Shape shape = residual->shape();

    if (dtype != Dtype::F32 && dtype != Dtype::F64)
        throw NotImplementedError("broyden_probe: dtype " + std::string(dtype_name(dtype)) +
                                  " is not supported; promote to float32 or float64 first");

    const diffeq::OperandSpec spec = diffeq::OperandSpec::from(residual, "broyden_probe");
    // The linear solve returns an (n, 1) column against a flat residual, so
    // only the element count has to agree.  Every operand here is reduced to a
    // scalar, so shape never enters the arithmetic.
    diffeq::check_operand(step, "broyden_probe.step", spec, diffeq::ShapeRule::SameCount);
    diffeq::check_operand(state, "broyden_probe.state", spec, diffeq::ShapeRule::SameCount);
    if (info && info->device() != device)
        throw DeviceMismatch(std::string(device_name(device)),
                             std::string(device_name(info->device())), "broyden_probe");

    OpScopeFull scope{"broyden_probe", device, dtype, shape};

    const std::size_t n = shape_numel(shape);

    // No gradient is wired: these are control flow, so a graph built here
    // would hold nodes nothing can consume.
    NoGradGuard no_grad;
    const TensorImplPtr res_c = residual->is_contiguous() ? residual : contiguous_op(residual);
    const TensorImplPtr step_c = step->is_contiguous() ? step : contiguous_op(step);
    const TensorImplPtr state_c = state->is_contiguous() ? state : contiguous_op(state);
    const TensorImplPtr info_c = (!info || info->is_contiguous()) ? info : contiguous_op(info);

    BroydenProbeResult out;

    if (device == Device::CPU) {
        const auto& rs = std::get<CpuStorage>(res_c->storage());
        const auto& ss = std::get<CpuStorage>(step_c->storage());
        const auto& xs = std::get<CpuStorage>(state_c->storage());
        if (dtype == Dtype::F64) {
            out.residual_sq = sum_squares(reinterpret_cast<const double*>(rs.ptr.get()), n);
            out.step_sq = sum_squares(reinterpret_cast<const double*>(ss.ptr.get()), n);
            out.state_sq = sum_squares(reinterpret_cast<const double*>(xs.ptr.get()), n);
        } else {
            out.residual_sq = sum_squares(reinterpret_cast<const float*>(rs.ptr.get()), n);
            out.step_sq = sum_squares(reinterpret_cast<const float*>(ss.ptr.get()), n);
            out.state_sq = sum_squares(reinterpret_cast<const float*>(xs.ptr.get()), n);
        }
        if (info_c)
            out.info = read_any_scalar(std::get<CpuStorage>(info_c->storage()), info_c->dtype());
        return out;
    }

    // GPU: one lazy expression covering every value, so the download below is
    // the only point at which anything evaluates.  Packing them into a single
    // array is what makes it one flush rather than one per question.
    namespace mx = ::mlx::core;
    const mx::array& r = *std::get<GpuStorage>(res_c->storage()).arr;
    const mx::array& s = *std::get<GpuStorage>(step_c->storage()).arr;
    const mx::array& x = *std::get<GpuStorage>(state_c->storage()).arr;
    const mx::Dtype mdt = r.dtype();

    // Reductions are taken over the flattened arrays so a column-shaped step
    // needs no reshape of its own.
    std::vector<mx::array> parts{mx::sum(mx::multiply(r, r)), mx::sum(mx::multiply(s, s)),
                                 mx::sum(mx::multiply(x, x))};
    if (info_c) {
        const mx::array& i = *std::get<GpuStorage>(info_c->storage()).arr;
        parts.push_back(mx::astype(mx::reshape(i, {}), mdt));
    }
    mx::array packed = mx::stack(parts);

    GpuStorage gs = gpu::wrap_mlx_array(std::move(packed), dtype);
    const CpuStorage host =
        gpu::download_gpu_to_cpu(gs, Shape{static_cast<std::int64_t>(parts.size())});

    if (dtype == Dtype::F64) {
        const auto* p = reinterpret_cast<const double*>(host.ptr.get());
        out.residual_sq = p[0];
        out.step_sq = p[1];
        out.state_sq = p[2];
        if (info_c)
            out.info = p[3];
    } else {
        const auto* p = reinterpret_cast<const float*>(host.ptr.get());
        out.residual_sq = static_cast<double>(p[0]);
        out.step_sq = static_cast<double>(p[1]);
        out.state_sq = static_cast<double>(p[2]);
        if (info_c)
            out.info = static_cast<double>(p[3]);
    }
    return out;
}

}  // namespace lucid
