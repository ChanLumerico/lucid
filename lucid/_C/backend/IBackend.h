#pragma once

// =====================================================================
// Lucid C++ engine — IBackend: pure virtual hardware backend interface.
// =====================================================================
//
// Phase 4: centralises all device-specific compute behind a single
// interface. Op kernels call `Dispatcher::for_device(d).method(...)`;
// neither the kernel nor the op file knows whether it runs on CPU or GPU.
//
// Adding a new backend (e.g. future CUDA / direct Metal) = implement
// IBackend; register with Dispatcher. Zero changes to any op file.
//
// Naming conventions:
//   - All methods take contiguous Storage inputs and return a new Storage.
//   - Shape / Dtype / Device parameters describe the *output* when
//     ambiguous; inputs always carry their own metadata inside Storage.
//   - Optional tensor inputs are passed as nullptr CpuStorage / nullptr
//     GpuStorage (caller checks before populating). Methods document
//     which inputs are optional.
//
// Layer: backend/. Depends on core/ only.

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {
namespace backend {

// -----------------------------------------------------------------------
// Options structs — thin POD bundles so individual methods stay concise.
// -----------------------------------------------------------------------

struct MatmulOpts {
    bool transA = false;
    bool transB = false;
    int M = 0, K = 0, N = 0;
    std::size_t batch = 1;
};

struct ConvOpts {
    int ndim = 2;  ///< 1, 2, or 3
    int N = 0, C_in = 0, C_out = 0;
    Shape input_shape;  ///< spatial dims only (H, W) or (L) or (D, H, W)
    Shape kernel_shape;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    int groups = 1;
    bool with_bias = false;
};

struct ReduceOpts {
    std::vector<int> axes;
    bool keepdims = false;
};

// -----------------------------------------------------------------------
// IBackend
// -----------------------------------------------------------------------

class IBackend {
public:
    virtual ~IBackend() = default;

    /// Which device this backend owns.
    virtual Device device() const noexcept = 0;

    // ---- Memory -------------------------------------------------------

    /// Allocate a zero-filled tensor.
    virtual Storage zeros(const Shape& shape, Dtype dt) = 0;

    /// Allocate a one-filled tensor.
    virtual Storage ones(const Shape& shape, Dtype dt) = 0;

    /// Clone a storage buffer (deep copy).
    virtual Storage clone(const Storage& src, const Shape& shape, Dtype dt) = 0;

    // ---- Elementwise binary -------------------------------------------
    // All elementwise binary ops expect pre-broadcast (same-shape) inputs.

    virtual Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // ---- Elementwise unary --------------------------------------------

    virtual Storage exp(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage log(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sqrt(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage rsqrt(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage abs(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage neg(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sign(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage floor(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage ceil(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage round(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sin(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cos(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage tanh(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sigmoid(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage relu(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // ---- Additional unary (Phase 4.5) ---------------------------------

    virtual Storage log2(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage reciprocal(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage square(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cube(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cube_root(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage tan(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage asin(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage acos(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage atan(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage sinh(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage cosh(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage invert(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage silu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage gelu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage softplus(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage selu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage mish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage relu6(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // ---- Reduction ----------------------------------------------------

    virtual Storage reduce_sum(const Storage& a,
                               const Shape& in_shape,
                               const ReduceOpts& opts,
                               Dtype dt) = 0;
    virtual Storage reduce_mean(const Storage& a,
                                const Shape& in_shape,
                                const ReduceOpts& opts,
                                Dtype dt) = 0;
    virtual Storage variance(const Storage& a,
                             const Shape& in_shape,
                             const ReduceOpts& opts,
                             Dtype dt) = 0;
    virtual Storage reduce_max(const Storage& a,
                               const Shape& in_shape,
                               const ReduceOpts& opts,
                               Dtype dt) = 0;
    virtual Storage reduce_min(const Storage& a,
                               const Shape& in_shape,
                               const ReduceOpts& opts,
                               Dtype dt) = 0;

    virtual Storage cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage softmax_backward(const Storage& z,
                                     const Storage& grad_out,
                                     const Shape& shape,
                                     int axis,
                                     Dtype dt) = 0;
    virtual Storage reverse_along_axis(const Storage& a,
                                       const Shape& shape,
                                       int axis,
                                       Dtype dt) = 0;
    virtual Storage trace(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage trace_backward(const Storage& grad_out,
                                   const Shape& input_shape,
                                   Dtype dt) = 0;
    virtual std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                          const Shape& out_shape,
                                          Dtype dt,
                                          bool indexing_xy) = 0;

    // ---- Linear algebra -----------------------------------------------

    /// N-D batched matrix multiply: a [...,M,K] @ b [...,K,N] → [...,M,N].
    virtual Storage matmul(const Storage& a,
                           const Storage& b,
                           const MatmulOpts& opts,
                           Dtype dt) = 0;

    // ---- Broadcast / cast --------------------------------------------

    virtual Storage broadcast(const Storage& a,
                              const Shape& src_shape,
                              const Shape& dst_shape,
                              Dtype dt) = 0;

    virtual Storage repeat(const Storage& a,
                           const Shape& shape,
                           Dtype dt,
                           std::int64_t repeats,
                           int axis) = 0;

    virtual Storage tile(const Storage& a,
                         const Shape& shape,
                         Dtype dt,
                         const std::vector<std::int64_t>& reps) = 0;

    virtual Storage permute(const Storage& a,
                            const Shape& shape,
                            const std::vector<int>& perm,
                            Dtype dt) = 0;

    virtual Storage pad(const Storage& a,
                        const Shape& shape,
                        Dtype dt,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        double constant) = 0;

    virtual Storage pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) = 0;
    virtual Storage rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) = 0;
    virtual Storage clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) = 0;

    virtual Storage cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) = 0;
};

}  // namespace backend
}  // namespace lucid
