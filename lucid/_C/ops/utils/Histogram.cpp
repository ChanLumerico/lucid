#include "Histogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <variant>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;

// Materialize an input on CPU regardless of source device.
CpuStorage to_cpu(const TensorImplPtr& a) {
    if (a->device_ == Device::GPU)
        return gpu::download_gpu_to_cpu(std::get<GpuStorage>(a->storage_), a->shape_);
    return std::get<CpuStorage>(a->storage_);
}

// Wrap a freshly-built CpuStorage as a Storage that lives on `target_device`,
// uploading to GPU when feasible so the result tracks the input's device.
// Histogram outputs are F64 (counts/density), which MLX-Metal does not
// support, so when the user requests GPU we fall back to CPU here and
// document the asymmetry — the alternative is silently downcasting to F32
// which loses precision.
Storage to_device_storage(CpuStorage&& cpu, Device target_device, const Shape& shape) {
    if (target_device == Device::GPU && cpu.dtype != Dtype::F64) {
        return Storage{gpu::upload_cpu_to_gpu(cpu, shape)};
    }
    return Storage{std::move(cpu)};
}

// Pick the device for histogram outputs: GPU only when the dtype can live
// on GPU (everything except F64).
Device pick_out_device(Device requested, Dtype dt) {
    if (requested == Device::GPU && dt == Dtype::F64)
        return Device::CPU;
    return requested;
}

// Read element `i` from a CPU storage as double.
double read_double(const CpuStorage& s, std::size_t i, Dtype dt) {
    switch (dt) {
        case Dtype::F32:
            return reinterpret_cast<const float*>(s.ptr.get())[i];
        case Dtype::F64:
            return reinterpret_cast<const double*>(s.ptr.get())[i];
        case Dtype::I32:
            return reinterpret_cast<const std::int32_t*>(s.ptr.get())[i];
        case Dtype::I64:
            return reinterpret_cast<const std::int64_t*>(s.ptr.get())[i];
        default:
            ErrorBuilder("histogram").not_implemented("dtype not supported");
    }
}

// Build linearly-spaced bin edges of length bins+1.
TensorImplPtr build_edges(double lo, double hi, std::int64_t bins) {
    Shape sh{bins + 1};
    auto cpu = allocate_cpu(sh, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(cpu.ptr.get());
    const double step = (hi - lo) / static_cast<double>(bins);
    for (std::int64_t i = 0; i <= bins; ++i)
        dst[i] = lo + static_cast<double>(i) * step;
    dst[bins] = hi;  // pin to exact endpoint
    return fresh(Storage{std::move(cpu)}, std::move(sh), Dtype::F64, Device::CPU);
}

}  // namespace

std::vector<TensorImplPtr> histogram_op(
    const TensorImplPtr& a, std::int64_t bins, double lo, double hi, bool density) {
    Validator::input(a, "histogram.a").non_null();
    if (bins <= 0)
        ErrorBuilder("histogram").fail("bins must be > 0");
    if (hi <= lo)
        ErrorBuilder("histogram").fail("hi must be > lo");
    OpScopeFull scope{"histogram", a->device_, a->dtype_, a->shape_};

    const auto cpu = to_cpu(a);
    const std::size_t n = shape_numel(a->shape_);
    const double step = (hi - lo) / static_cast<double>(bins);

    Shape counts_shape{bins};
    auto counts = allocate_cpu(counts_shape, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(counts.ptr.get());
    std::memset(dst, 0, counts.nbytes);

    for (std::size_t i = 0; i < n; ++i) {
        const double v = read_double(cpu, i, a->dtype_);
        if (v < lo || v > hi)
            continue;
        std::int64_t bin = static_cast<std::int64_t>((v - lo) / step);
        if (bin >= bins)
            bin = bins - 1;  // include right edge
        dst[bin] += 1.0;
    }

    if (density) {
        for (std::int64_t i = 0; i < bins; ++i)
            dst[i] /= (n * step);
    }

    Device out_dev = pick_out_device(a->device_, Dtype::F64);
    auto counts_storage = to_device_storage(std::move(counts), out_dev, counts_shape);
    auto counts_t = fresh(std::move(counts_storage), std::move(counts_shape), Dtype::F64, out_dev);
    auto edges = build_edges(lo, hi, bins);
    // edges share the dtype/device convention with counts.
    if (out_dev == Device::GPU) {
        edges = fresh(
            Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(edges->storage_), edges->shape_)},
            edges->shape_, Dtype::F64, Device::GPU);
    }
    return {counts_t, edges};
}

std::vector<TensorImplPtr> histogram2d_op(const TensorImplPtr& a,
                                          const TensorImplPtr& b,
                                          std::int64_t bins_a,
                                          std::int64_t bins_b,
                                          double lo_a,
                                          double hi_a,
                                          double lo_b,
                                          double hi_b,
                                          bool density) {
    if (!a || !b)
        ErrorBuilder("histogram2d").fail("null input");
    if (a->shape_ != b->shape_)
        throw ShapeMismatch(a->shape_, b->shape_, "histogram2d");
    if (bins_a <= 0 || bins_b <= 0)
        ErrorBuilder("histogram2d").fail("bins must be > 0");
    OpScopeFull scope{"histogram2d", a->device_, a->dtype_, a->shape_};

    const auto ca = to_cpu(a);
    const auto cb = to_cpu(b);
    const std::size_t n = shape_numel(a->shape_);
    const double step_a = (hi_a - lo_a) / static_cast<double>(bins_a);
    const double step_b = (hi_b - lo_b) / static_cast<double>(bins_b);

    Shape counts_shape{bins_a, bins_b};
    auto counts = allocate_cpu(counts_shape, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(counts.ptr.get());
    std::memset(dst, 0, counts.nbytes);

    for (std::size_t i = 0; i < n; ++i) {
        const double va = read_double(ca, i, a->dtype_);
        const double vb = read_double(cb, i, b->dtype_);
        if (va < lo_a || va > hi_a)
            continue;
        if (vb < lo_b || vb > hi_b)
            continue;
        std::int64_t ia = static_cast<std::int64_t>((va - lo_a) / step_a);
        std::int64_t ib = static_cast<std::int64_t>((vb - lo_b) / step_b);
        if (ia >= bins_a)
            ia = bins_a - 1;
        if (ib >= bins_b)
            ib = bins_b - 1;
        dst[ia * bins_b + ib] += 1.0;
    }

    if (density) {
        const double area = step_a * step_b * n;
        for (std::size_t i = 0; i < (std::size_t)(bins_a * bins_b); ++i)
            dst[i] /= area;
    }

    Device out_dev = pick_out_device(a->device_, Dtype::F64);
    auto counts_storage = to_device_storage(std::move(counts), out_dev, counts_shape);
    auto counts_t = fresh(std::move(counts_storage), std::move(counts_shape), Dtype::F64, out_dev);
    // Pack edges as a (bins_a + bins_b + 2) flat tensor: [edges_a..., edges_b...]
    auto ea = build_edges(lo_a, hi_a, bins_a);
    auto eb = build_edges(lo_b, hi_b, bins_b);
    Shape edge_shape{bins_a + 1 + bins_b + 1};
    auto edges = allocate_cpu(edge_shape, Dtype::F64);
    auto* edst = reinterpret_cast<double*>(edges.ptr.get());
    std::memcpy(edst, std::get<CpuStorage>(ea->storage_).ptr.get(), (bins_a + 1) * sizeof(double));
    std::memcpy(edst + (bins_a + 1), std::get<CpuStorage>(eb->storage_).ptr.get(),
                (bins_b + 1) * sizeof(double));
    auto edges_storage = to_device_storage(std::move(edges), out_dev, edge_shape);
    auto edges_t = fresh(std::move(edges_storage), std::move(edge_shape), Dtype::F64, out_dev);
    return {counts_t, edges_t};
}

std::vector<TensorImplPtr> histogramdd_op(const TensorImplPtr& a,
                                          std::vector<std::int64_t> bins,
                                          std::vector<std::pair<double, double>> ranges,
                                          bool density) {
    Validator::input(a, "histogramdd.a").non_null();
    if (a->shape_.size() != 2)
        ErrorBuilder("histogramdd").fail("input must be 2-D (N, D)");
    const std::int64_t N = a->shape_[0];
    const std::int64_t D = a->shape_[1];
    if ((std::int64_t)bins.size() != D || (std::int64_t)ranges.size() != D)
        ErrorBuilder("histogramdd").fail("bins/ranges length must equal D");
    OpScopeFull scope{"histogramdd", a->device_, a->dtype_, a->shape_};

    const auto ca = to_cpu(a);

    std::vector<double> step(D);
    for (std::int64_t d = 0; d < D; ++d) {
        if (ranges[d].second <= ranges[d].first)
            ErrorBuilder("histogramdd").fail("each range hi must be > lo");
        step[d] = (ranges[d].second - ranges[d].first) / static_cast<double>(bins[d]);
    }

    Shape counts_shape(bins.begin(), bins.end());
    auto counts = allocate_cpu(counts_shape, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(counts.ptr.get());
    std::memset(dst, 0, counts.nbytes);

    Stride stride(D);
    if (D > 0) {
        stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)D - 2; d >= 0; --d)
            stride[d] = stride[d + 1] * bins[d + 1];
    }

    for (std::int64_t i = 0; i < N; ++i) {
        std::size_t flat = 0;
        bool inside = true;
        for (std::int64_t d = 0; d < D; ++d) {
            const double v = read_double(ca, i * D + d, a->dtype_);
            if (v < ranges[d].first || v > ranges[d].second) {
                inside = false;
                break;
            }
            std::int64_t bd = static_cast<std::int64_t>((v - ranges[d].first) / step[d]);
            if (bd >= bins[d])
                bd = bins[d] - 1;
            flat += static_cast<std::size_t>(bd) * static_cast<std::size_t>(stride[d]);
        }
        if (inside)
            dst[flat] += 1.0;
    }

    if (density) {
        double cell_volume = 1.0;
        for (auto s : step)
            cell_volume *= s;
        const double total = cell_volume * static_cast<double>(N);
        const std::size_t total_cells = shape_numel(counts_shape);
        for (std::size_t i = 0; i < total_cells; ++i)
            dst[i] /= total;
    }

    Device out_dev = pick_out_device(a->device_, Dtype::F64);
    auto counts_storage = to_device_storage(std::move(counts), out_dev, counts_shape);
    auto counts_t = fresh(std::move(counts_storage), std::move(counts_shape), Dtype::F64, out_dev);

    // Pack all edges sequentially.
    std::int64_t edge_total = 0;
    for (auto b_ : bins)
        edge_total += (b_ + 1);
    Shape edge_shape{edge_total};
    auto edges = allocate_cpu(edge_shape, Dtype::F64);
    auto* edst = reinterpret_cast<double*>(edges.ptr.get());
    std::int64_t off = 0;
    for (std::int64_t d = 0; d < D; ++d) {
        const double lo = ranges[d].first;
        const double hi = ranges[d].second;
        for (std::int64_t i = 0; i <= bins[d]; ++i)
            edst[off + i] = lo + static_cast<double>(i) * step[d];
        edst[off + bins[d]] = hi;
        off += bins[d] + 1;
    }
    auto edges_storage = to_device_storage(std::move(edges), out_dev, edge_shape);
    auto edges_t = fresh(std::move(edges_storage), std::move(edge_shape), Dtype::F64, out_dev);
    return {counts_t, edges_t};
}

}  // namespace lucid
