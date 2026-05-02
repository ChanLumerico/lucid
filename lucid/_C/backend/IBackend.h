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

struct ClassLossForwardResult {
    Storage output;
    Storage saved_aux;
    Storage valid_count;
};

struct StoragePair {
    Storage first;
    Storage second;
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

    /// Materialize a logical tensor view into contiguous storage.
    virtual Storage contiguous(const Storage& src,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset,
                               bool already_contiguous,
                               Dtype dt) = 0;

    // ---- Elementwise binary -------------------------------------------
    // All elementwise binary ops expect pre-broadcast (same-shape) inputs.

    virtual Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage bitwise_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;
    virtual Storage compare_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;
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
    virtual Storage gelu_backward(const Storage& a,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt) = 0;
    virtual Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) = 0;
    virtual Storage softplus(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) = 0;
    virtual Storage elu_backward(
        const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) = 0;
    virtual Storage selu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage selu_backward(const Storage& a,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt) = 0;
    virtual Storage mish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage mish_backward(const Storage& a,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt) = 0;
    virtual Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_sigmoid_backward(const Storage& a,
                                          const Storage& grad,
                                          const Shape& shape,
                                          Dtype dt) = 0;
    virtual Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_swish_backward(const Storage& a,
                                        const Storage& grad,
                                        const Shape& shape,
                                        Dtype dt) = 0;
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
    virtual Storage softmax_backward(
        const Storage& z, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage reverse_along_axis(const Storage& a,
                                       const Shape& shape,
                                       int axis,
                                       Dtype dt) = 0;
    virtual Storage trace(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) = 0;
    virtual std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                          const Shape& out_shape,
                                          Dtype dt,
                                          bool indexing_xy) = 0;
    virtual Storage where_branch(const Storage& grad,
                                 const Storage& cond,
                                 const Shape& shape,
                                 Dtype dt,
                                 bool true_branch) = 0;
    virtual Storage masked_fill(
        const Storage& a, const Storage& mask, const Shape& shape, Dtype dt, double value) = 0;
    virtual Storage gather(const Storage& a,
                           const Storage& indices,
                           const Shape& input_shape,
                           const Shape& output_shape,
                           int axis,
                           Dtype index_dtype,
                           Dtype dt) = 0;
    virtual Storage gather_backward(const Storage& grad,
                                    const Storage& indices,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    Dtype index_dtype,
                                    Dtype dt) = 0;
    virtual Storage diagonal(
        const Storage& a, const Shape& input_shape, int offset, int axis1, int axis2, Dtype dt) = 0;
    virtual Storage diagonal_backward(const Storage& grad,
                                      const Shape& input_shape,
                                      const Shape& output_shape,
                                      int offset,
                                      int axis1,
                                      int axis2,
                                      Dtype dt) = 0;
    virtual Storage roll(const Storage& a,
                         const Shape& shape,
                         Dtype dt,
                         const std::vector<std::int64_t>& shifts,
                         const std::vector<int>& axes) = 0;
    virtual Storage reshape(const Storage& a,
                            const Shape& src_shape,
                            const Shape& dst_shape,
                            Dtype dt) = 0;
    virtual Storage slice_axis(const Storage& a,
                               const Shape& src_shape,
                               const Shape& slice_shape,
                               int axis,
                               std::int64_t offset,
                               Dtype dt) = 0;
    virtual Storage insert_axis_slice(const Storage& a,
                                      const Shape& src_shape,
                                      const Shape& dst_shape,
                                      int axis,
                                      std::int64_t offset,
                                      Dtype dt) = 0;
    virtual Storage concatenate(const std::vector<Storage>& xs,
                                const std::vector<Shape>& shapes,
                                int axis,
                                Dtype dt) = 0;
    virtual Storage stack(const std::vector<Storage>& xs,
                          const Shape& input_shape,
                          int axis,
                          Dtype dt) = 0;
    virtual std::vector<Storage> split_equal(
        const Storage& a, const Shape& shape, int axis, std::int64_t num_splits, Dtype dt) = 0;
    virtual std::vector<Storage> split_at(const Storage& a,
                                          const Shape& shape,
                                          int axis,
                                          const std::vector<std::int64_t>& indices,
                                          Dtype dt) = 0;
    virtual Storage repeat_backward(const Storage& grad_out,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    std::int64_t repeats,
                                    Dtype dt) = 0;
    virtual Storage tile_backward(const Storage& grad_out,
                                  const Shape& input_shape,
                                  const Shape& padded_shape,
                                  const Shape& output_shape,
                                  const std::vector<std::int64_t>& reps,
                                  Dtype dt) = 0;
    virtual std::pair<Storage, Storage> sort_select(const Storage& a,
                                                    const Shape& input_shape,
                                                    const Shape& output_shape,
                                                    int axis,
                                                    Dtype dt,
                                                    bool descending) = 0;
    virtual Storage argsort(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage arg_reduce_index(
        const Storage& a, const Shape& shape, int axis, bool keepdims, Dtype dt, bool is_min) = 0;
    virtual Storage scatter_add_axis(const Storage& grad,
                                     const Storage& indices,
                                     const Shape& output_shape,
                                     const Shape& grad_shape,
                                     int axis,
                                     Dtype dt) = 0;

    // ---- Linear algebra -----------------------------------------------

    /// N-D batched matrix multiply: a [...,M,K] @ b [...,K,N] → [...,M,N].
    virtual Storage matmul(const Storage& a,
                           const Storage& b,
                           const MatmulOpts& opts,
                           Dtype dt) = 0;
    virtual Storage linear(const Storage& x,
                           const Storage& weight,
                           const Storage& bias,
                           const Shape& x_shape,
                           const Shape& weight_shape,
                           const Shape& out_shape,
                           Dtype dt) = 0;
    virtual std::vector<Storage> linear_backward(const Storage& grad,
                                                 const Storage& x,
                                                 const Storage& weight,
                                                 const Shape& x_shape,
                                                 const Shape& weight_shape,
                                                 const Shape& bias_shape,
                                                 Dtype dt) = 0;
    virtual StoragePair rms_norm_forward(const Storage& x,
                                         const Storage& gamma,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         double eps,
                                         const Shape& x_shape,
                                         Dtype dt) = 0;
    virtual StoragePair rms_norm_backward(const Storage& x,
                                          const Storage& gamma,
                                          const Storage& saved_rstd,
                                          const Storage& grad,
                                          std::size_t outer,
                                          std::size_t normalized_size,
                                          const Shape& x_shape,
                                          const Shape& gamma_shape,
                                          Dtype dt) = 0;
    virtual std::vector<Storage> layer_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    std::size_t outer,
                                                    std::size_t normalized_size,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;
    virtual std::vector<Storage> layer_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     std::size_t outer,
                                                     std::size_t normalized_size,
                                                     const Shape& x_shape,
                                                     const Shape& gamma_shape,
                                                     const Shape& beta_shape,
                                                     Dtype dt) = 0;
    virtual std::vector<Storage> batch_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int spatial,
                                                    int ndim,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;
    virtual std::vector<Storage> batch_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int spatial,
                                                     int ndim,
                                                     const Shape& x_shape,
                                                     Dtype dt) = 0;
    virtual std::vector<Storage> group_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    int batch,
                                                    int channels,
                                                    int spatial,
                                                    int groups,
                                                    const std::vector<int>& spatial_dims,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;
    virtual std::vector<Storage> group_norm_backward(const Storage& x,
                                                     const Storage& gamma,
                                                     const Storage& saved_mean,
                                                     const Storage& saved_rstd,
                                                     const Storage& grad,
                                                     int batch,
                                                     int channels,
                                                     int spatial,
                                                     int groups,
                                                     const std::vector<int>& spatial_dims,
                                                     const Shape& x_shape,
                                                     Dtype dt) = 0;
    virtual Storage linalg_norm(const Storage& a,
                                const Shape& shape,
                                double ord,
                                const std::vector<int>& axes,
                                bool keepdims,
                                Dtype dt) = 0;
    virtual Storage linalg_cholesky(const Storage& a, const Shape& shape, bool upper, Dtype dt) = 0;
    virtual Storage linalg_inv(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage linalg_solve(const Storage& a,
                                 const Storage& b,
                                 const Shape& a_shape,
                                 const Shape& b_shape,
                                 Dtype dt) = 0;
    virtual Storage linalg_matrix_power(const Storage& a,
                                        const Shape& shape,
                                        int power,
                                        Dtype dt) = 0;
    virtual Storage linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual StoragePair linalg_qr(const Storage& a,
                                  const Shape& shape,
                                  const Shape& q_shape,
                                  const Shape& r_shape,
                                  Dtype dt) = 0;
    virtual StoragePair linalg_eig(const Storage& a,
                                   const Shape& shape,
                                   const Shape& values_shape,
                                   const Shape& vectors_shape,
                                   Dtype dt) = 0;
    virtual std::vector<Storage> linalg_svd(const Storage& a,
                                            const Shape& shape,
                                            bool compute_uv,
                                            const Shape& u_shape,
                                            const Shape& s_shape,
                                            const Shape& vt_shape,
                                            Dtype dt) = 0;

    // ---- Broadcast / cast --------------------------------------------

    virtual Storage broadcast(const Storage& a,
                              const Shape& src_shape,
                              const Shape& dst_shape,
                              Dtype dt) = 0;

    virtual Storage repeat(
        const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) = 0;

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
    virtual Storage clip(
        const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) = 0;

    virtual Storage cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) = 0;

    virtual Storage mse_loss(const Storage& input,
                             const Storage& target,
                             const Shape& shape,
                             Dtype dt,
                             int reduction) = 0;
    virtual std::pair<Storage, Storage> mse_loss_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) = 0;
    virtual Storage huber_loss(const Storage& input,
                               const Storage& target,
                               const Shape& shape,
                               Dtype dt,
                               double delta,
                               int reduction) = 0;
    virtual std::pair<Storage, Storage> huber_loss_backward(const Storage& input,
                                                            const Storage& target,
                                                            const Storage& grad,
                                                            const Shape& shape,
                                                            Dtype dt,
                                                            double delta,
                                                            int reduction) = 0;
    virtual Storage bce_loss(const Storage& input,
                             const Storage& target,
                             const Storage& weight,
                             const Shape& shape,
                             Dtype dt,
                             double eps,
                             int reduction) = 0;
    virtual std::vector<Storage> bce_loss_backward(const Storage& input,
                                                   const Storage& target,
                                                   const Storage& weight,
                                                   const Storage& grad,
                                                   const Shape& shape,
                                                   Dtype dt,
                                                   double eps,
                                                   int reduction) = 0;
    virtual Storage bce_with_logits_loss(const Storage& input,
                                         const Storage& target,
                                         const Storage& weight,
                                         const Storage& pos_weight,
                                         const Shape& shape,
                                         const Shape& weight_shape,
                                         const Shape& pos_weight_shape,
                                         Dtype dt,
                                         int reduction) = 0;
    virtual std::vector<Storage> bce_with_logits_backward(const Storage& input,
                                                          const Storage& target,
                                                          const Storage& weight,
                                                          const Storage& pos_weight,
                                                          const Storage& grad,
                                                          const Shape& shape,
                                                          Dtype dt,
                                                          int reduction) = 0;

    virtual ClassLossForwardResult cross_entropy_loss(const Storage& input,
                                                      const Storage& target,
                                                      const Storage* weight,
                                                      const Shape& input_shape,
                                                      const Shape& target_shape,
                                                      Dtype dt,
                                                      double eps,
                                                      int ignore_index,
                                                      int reduction) = 0;
    virtual Storage cross_entropy_backward(const Storage& saved_softmax,
                                           const Storage& target,
                                           const Storage* weight,
                                           const Storage& valid_count,
                                           const Storage& grad,
                                           const Shape& input_shape,
                                           Dtype dt,
                                           int ignore_index,
                                           int reduction) = 0;
    virtual ClassLossForwardResult nll_loss(const Storage& input,
                                            const Storage& target,
                                            const Storage* weight,
                                            const Shape& input_shape,
                                            const Shape& target_shape,
                                            Dtype dt,
                                            int ignore_index,
                                            int reduction) = 0;
    virtual Storage nll_loss_backward(const Storage& target,
                                      const Storage* weight,
                                      const Storage& valid_count,
                                      const Storage& grad,
                                      const Shape& input_shape,
                                      Dtype dt,
                                      int ignore_index,
                                      int reduction) = 0;

    virtual CpuStorage to_cpu(const Storage& a, const Shape& shape) = 0;

    // ---- Attention --------------------------------------------------------

    /// Scaled dot-product attention forward.
    /// Returns {output_storage, weights_storage}.
    /// attn_mask may be nullptr (no mask). mask_dtype is only meaningful when
    /// attn_mask != nullptr.
    virtual std::vector<Storage> sdpa_forward(
        const Storage& q,
        const Storage& k,
        const Storage& v,
        const Storage* attn_mask,  // nullptr if no mask
        const Shape& q_shape,
        const Shape& k_shape,
        const Shape& v_shape,
        Dtype mask_dtype,          // dtype of attn_mask; ignored when attn_mask==nullptr
        std::size_t mask_numel,    // numel of mask; ignored when attn_mask==nullptr
        double scale,
        bool is_causal,
        Dtype dt) = 0;

    /// Scaled dot-product attention backward.
    /// Returns {dQ, dK, dV}.
    virtual std::vector<Storage> sdpa_backward(
        const Storage& grad_out,
        const Storage& q,
        const Storage& k,
        const Storage& v,
        const Storage& saved_weights,
        const Shape& q_shape,
        const Shape& k_shape,
        const Shape& v_shape,
        double scale,
        Dtype dt) = 0;

    // ---- Transposed convolution ------------------------------------------

    /// N-D transposed convolution forward (N = 1, 2, or 3).
    /// stride/pad/opad/kernel_shape are length-N arrays; dilation is currently
    /// always 1 and reserved for future use.
    virtual Storage conv_transpose_nd_forward(
        const Storage& x,
        const Storage& W,
        const Storage& b,
        int B,
        int Cin,
        int Cout,
        const int* S,   // input spatial dims, length N
        const int* K,   // kernel spatial dims, length N
        const int* O,   // output spatial dims, length N
        const int* stride,
        const int* pad,
        const int* opad,
        int N,
        const Shape& out_shape,
        Dtype dt) = 0;

    /// N-D transposed convolution backward.
    /// Returns {dx, dW, db}.
    virtual std::vector<Storage> conv_transpose_nd_backward(
        const Storage& grad_out,
        const Storage& x,
        const Storage& W,
        int B,
        int Cin,
        int Cout,
        const int* S,
        const int* K,
        const int* O,
        const int* stride,
        const int* pad,
        int N,
        Dtype dt) = 0;

    // ---- Convolution (N-D) -------------------------------------------

    struct ConvNdOpts {
        int N;            ///< spatial dims (1, 2, or 3)
        int groups;
        int stride[3];    ///< length N, rest 0
        int pad[3];       ///< length N
        int dilation[3];  ///< length N
    };

    /// N-D convolution forward: x[B,Cin,S...] * W[Cout,Cin/g,K...] + b[Cout] -> out[B,Cout,O...]
    virtual Storage conv_nd_forward(
        const Storage& x, const Storage& W, const Storage& b,
        int B, int Cin, int Cout, int Cin_g, int Cout_g,
        const int* S, const int* K, const int* O,
        const ConvNdOpts& opts,
        const Shape& out_shape, Dtype dt) = 0;

    /// Backward for conv_nd: grad_out -> {dx[B,Cin,S...], dW[Cout,Cin/g,K...], db[Cout]}
    virtual std::vector<Storage> conv_nd_backward(
        const Storage& grad_out, const Storage& x, const Storage& W,
        int B, int Cin, int Cout, int Cin_g, int Cout_g,
        const int* S, const int* K, const int* O,
        const ConvNdOpts& opts, Dtype dt) = 0;

    /// Unfold forward: x[B,C,S...] -> out[B, C*K_total, O_total]
    virtual Storage unfold_forward(
        const Storage& x,
        int B, int C,
        const std::vector<int>& S, const std::vector<int>& K, const std::vector<int>& O,
        const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation,
        const Shape& out_shape, Dtype dt) = 0;

    /// Unfold backward: grad_out[B, C*K_total, O_total] -> dx[B,C,S...]
    virtual Storage unfold_backward(
        const Storage& grad_out,
        int B, int C,
        const std::vector<int>& S, const std::vector<int>& K, const std::vector<int>& O,
        const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation,
        Dtype dt) = 0;

    // ---- Dropout helpers ---------------------------------------------

    /// Expand a (B,C,1,...,1) or (B,1,...,1) mask to full x_shape and multiply element-wise.
    /// Returns {expanded_mask, y}.
    virtual std::pair<Storage, Storage> expand_and_multiply(
        const Storage& mask, const Storage& x,
        const Shape& mask_shape, const Shape& x_shape, Dtype dt) = 0;

    /// DropBlock forward: given a pre-sampled Bernoulli seed storage (flat, shape x_shape),
    /// apply spatial dilation (block_size x block_size max-pool), compute keep = scale*(1-dilated),
    /// and multiply with x. Returns keep_mask (same shape as x).
    virtual Storage drop_block_mask(
        const Storage& seed, double drop_prob, std::int64_t block_size,
        const Shape& x_shape, Dtype dt) = 0;

    // ---- Embedding -------------------------------------------------------

    /// Embedding forward: gather rows of weight table indexed by indices.
    /// Returns output storage of shape out_shape (indices_shape ++ [D]).
    virtual Storage embedding_forward(const Storage& weight,
                                      const Storage& indices,
                                      const Shape& weight_shape,
                                      const Shape& indices_shape,
                                      const Shape& out_shape,
                                      int padding_idx,
                                      Dtype dt) = 0;

    /// Embedding backward: scatter-add gradients to weight rows.
    /// Returns dW storage of shape weight_shape.
    virtual Storage embedding_backward(const Storage& grad_out,
                                       const Storage& indices,
                                       const Shape& weight_shape,
                                       const Shape& indices_shape,
                                       int padding_idx,
                                       Dtype dt) = 0;

    /// Sinusoidal positional embedding: shape [seq_len, embed_dim].
    virtual Storage sinusoidal_pos_embedding(std::int64_t seq_len,
                                             std::int64_t embed_dim,
                                             Dtype dt) = 0;

    /// RoPE forward.  position_ids is a 1-D index storage or nullptr (sequential).
    /// Returns {out, saved_cos, saved_sin}.
    virtual std::vector<Storage> rope_forward(const Storage& x,
                                              const Storage* position_ids,
                                              const Shape& x_shape,
                                              bool interleaved,
                                              Dtype pos_dtype,
                                              Dtype dt) = 0;

    /// RoPE backward.  Returns dx.
    virtual Storage rope_backward(const Storage& grad_out,
                                  const Storage& saved_cos,
                                  const Storage& saved_sin,
                                  const Shape& x_shape,
                                  bool interleaved,
                                  Dtype dt) = 0;

    // ---- Pooling ---------------------------------------------------------

    struct PoolOpts {
        int K[3];      ///< kernel sizes (unused dims are 0)
        int stride[3]; ///< effective strides (after default-stride resolution)
        int pad[3];    ///< padding
        int N;         ///< spatial dims (1, 2, or 3)
    };

    /// Max pool forward.  Returns {output, argmax_indices (I32)}.
    virtual std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                                     const Shape& x_shape,
                                                     const Shape& out_shape,
                                                     const PoolOpts& opts,
                                                     Dtype dt) = 0;

    /// Max pool backward.  Returns dx.
    virtual Storage max_pool_nd_backward(const Storage& grad_out,
                                         const Storage& saved_argmax,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    /// Average pool forward.  Returns output.
    virtual Storage avg_pool_nd_forward(const Storage& x,
                                        const Shape& x_shape,
                                        const Shape& out_shape,
                                        const PoolOpts& opts,
                                        Dtype dt) = 0;

    /// Average pool backward.  Returns dx.
    virtual Storage avg_pool_nd_backward(const Storage& grad_out,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    // ---- BatchNorm eval (inference mode) ---------------------------------

    /// Forward: out = gamma*(x-mean)*rsqrt(var+eps)+beta. Returns {out, rstd}.
    virtual std::vector<Storage> batch_norm_eval_forward(
        const Storage& x, const Storage& mean, const Storage& var,
        const Storage& gamma, const Storage& beta,
        const Shape& x_shape, int C, int spatial, double eps, Dtype dt) = 0;

    /// Backward. Returns {dx, dmean, dvar, dgamma, dbeta}.
    virtual std::vector<Storage> batch_norm_eval_backward(
        const Storage& x, const Storage& mean, const Storage& gamma,
        const Storage& rstd, const Storage& grad_out,
        const Shape& x_shape, int C, int spatial, Dtype dt) = 0;

    // ---- Lp-normalize ----------------------------------------------------

    /// Lp-normalize along axis. Returns {out, saved_norm}.
    virtual std::vector<Storage> lp_normalize_forward(
        const Storage& x, const Shape& x_shape,
        double ord, int axis, double eps, Dtype dt) = 0;

    /// Backward. Returns dx.
    virtual Storage lp_normalize_backward(
        const Storage& x, const Storage& saved_norm, const Storage& grad_out,
        const Shape& x_shape, double ord, int axis, Dtype dt) = 0;

    // ---- Global response normalization (GRN, ConvNeXt-v2) ---------------

    /// Forward. Returns {out, saved_sq_mean, saved_gx}.
    virtual std::vector<Storage> global_response_norm_forward(
        const Storage& x, const Storage& gamma, const Storage& beta,
        const Shape& x_shape, double eps, Dtype dt) = 0;

    /// Backward. Returns {dx, dgamma, dbeta}.
    virtual std::vector<Storage> global_response_norm_backward(
        const Storage& x, const Storage& gamma, const Storage& beta,
        const Storage& saved_Nx, const Storage& grad_out,
        const Shape& x_shape, double eps, Dtype dt) = 0;

    // ---- Interpolation ---------------------------------------------------

    // ---- Nearest-neighbor interpolation (no grad) ------------------------------

    /// 2D nearest-neighbor upsample/downsample forward. No autograd.
    virtual Storage interpolate_nearest_2d_forward(
        const Storage& input, const Shape& in_shape,
        int H_out, int W_out, Dtype dt) = 0;

    /// 3D nearest-neighbor upsample/downsample forward. No autograd.
    virtual Storage interpolate_nearest_3d_forward(
        const Storage& input, const Shape& in_shape,
        int D_out, int H_out, int W_out, Dtype dt) = 0;

    /// Bilinear 2D interpolation forward.
    virtual Storage interpolate_bilinear_forward(
        const Storage& input, const Shape& in_shape,
        int H_out, int W_out, bool align_corners, Dtype dt) = 0;

    /// Bilinear 2D interpolation backward. Returns grad_input.
    virtual Storage interpolate_bilinear_backward(
        const Storage& grad_out, const Shape& in_shape,
        int H_out, int W_out, bool align_corners, Dtype dt) = 0;

    /// Trilinear 3D interpolation forward.
    virtual Storage interpolate_trilinear_forward(
        const Storage& input, const Shape& in_shape,
        int D_out, int H_out, int W_out, bool align_corners, Dtype dt) = 0;

    /// Trilinear 3D interpolation backward. Returns grad_input.
    virtual Storage interpolate_trilinear_backward(
        const Storage& grad_out, const Shape& in_shape,
        int D_out, int H_out, int W_out, bool align_corners, Dtype dt) = 0;

    // ---- Spatial transforms ---------------------------------------------

    /// Affine grid forward: theta [N,2,3] → grid [N,H,W,2].
    virtual Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W,
        bool align_corners, Dtype dt) = 0;

    /// Affine grid backward. Returns d_theta.
    virtual Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W,
        bool align_corners, Dtype dt) = 0;

    /// Grid sample forward (bilinear, mode/padding_mode flags).
    virtual Storage grid_sample_forward(
        const Storage& input, const Storage& grid,
        const Shape& in_shape, const Shape& grid_shape,
        int mode, int padding_mode, bool align_corners, Dtype dt) = 0;

    /// Grid sample backward. Returns {d_input, d_grid}.
    virtual std::vector<Storage> grid_sample_backward(
        const Storage& grad_out, const Storage& input, const Storage& grid,
        const Shape& in_shape, const Shape& grid_shape,
        int mode, int padding_mode, bool align_corners, Dtype dt) = 0;

    // ---- Vision ---------------------------------------------------------

    /// Bilinear layer forward: y = x1 @ W @ x2^T + b. Returns out.
    virtual Storage bilinear_layer_forward(
        const Storage& x1, const Storage& x2, const Storage& weight,
        const Storage& bias, bool has_bias,
        const Shape& x1_shape, const Shape& x2_shape, const Shape& w_shape,
        Dtype dt) = 0;

    /// Bilinear layer backward. Returns {dx1, dx2, dW, db}.
    virtual std::vector<Storage> bilinear_layer_backward(
        const Storage& grad_out, const Storage& x1, const Storage& x2,
        const Storage& weight,
        const Shape& x1_shape, const Shape& x2_shape, const Shape& w_shape,
        bool has_bias, Dtype dt) = 0;

    // ---- One-hot encoding -----------------------------------------------

    /// Forward: returns out[...shape..., num_classes] of type out_dtype.
    virtual Storage one_hot_forward(const Storage& indices,
                                    const Shape& indices_shape,
                                    int num_classes,
                                    Dtype out_dtype) = 0;

    // ---- Rotate (forward only) ------------------------------------------

    /// Forward: nearest-neighbor rotation. angle_rad_neg = -angle_deg*(π/180).
    virtual Storage rotate_forward(const Storage& input,
                                   const Shape& shape,
                                   double angle_rad_neg,
                                   double cx,
                                   double cy,
                                   Dtype dt) = 0;
};

}  // namespace backend
}  // namespace lucid
