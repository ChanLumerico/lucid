// lucid/_C/backend/IBackend.h
//
// Pure-virtual interface that every hardware backend must implement.  All
// compute operations in the Lucid ML engine — elementwise math, reductions,
// matrix multiply, convolution, normalization, pooling, loss functions, SDPA,
// LSTM, and more — are declared here as abstract virtual methods.  Higher-
// level tensor code calls through the Dispatcher to whichever concrete backend
// is registered for a given Device.  Adding a new backend means subclassing
// IBackend and registering an instance with Dispatcher::register_backend().

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/ErrorBuilder.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {
namespace backend {

// Parameter bag for a single matrix-multiplication call.
//
// All dimensions are in logical (not transposed) order: the product is
// (M x K) @ (K x N) = (M x N).  batch > 1 indicates a batched matmul where
// the same A/B strides repeat batch times.
struct MatmulOpts {
    bool transA = false;
    bool transB = false;
    int M = 0, K = 0, N = 0;
    std::size_t batch = 1;
};

// Parameter bag for a convolution op (used by some legacy paths).
//
// The primary convolution paths use ConvNdOpts and explicit dimension
// arguments; this struct is kept for compatibility.
struct ConvOpts {
    int ndim = 2;
    int N = 0, C_in = 0, C_out = 0;
    Shape input_shape;
    Shape kernel_shape;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> dilation;
    int groups = 1;
    bool with_bias = false;
};

// Parameter bag for axis-reduction operations.
//
// axes is the list of dimensions to reduce over (may be negative).  If axes
// is empty the caller should interpret that as "reduce all axes".  keepdims
// controls whether collapsed dimensions are kept as size-1.
struct ReduceOpts {
    std::vector<int> axes;
    bool keepdims = false;
};

// Aggregated return value from classification-loss forward passes.
//
// output is the scalar or per-sample loss.  saved_aux holds intermediate
// values needed by the backward pass (e.g. log-softmax activations for
// cross-entropy).  valid_count records the number of un-ignored samples so
// that mean reduction can be computed correctly.
struct ClassLossForwardResult {
    Storage output;
    Storage saved_aux;
    Storage valid_count;
};

// Convenience wrapper that bundles two Storage objects returned from ops
// that produce a pair of outputs (e.g. QR decomposition, sort+argsort).
struct StoragePair {
    Storage first;
    Storage second;
};

// Abstract compute backend interface.
//
// Ownership: each concrete backend is created once and stored inside the
// Dispatcher singleton.  The Dispatcher owns the backends via unique_ptr, so
// the lifetime of every IBackend equals the lifetime of the process.
//
// Thread safety: methods are not internally synchronized.  Callers that share
// a backend across threads must provide their own serialization, or ensure
// that the underlying framework (e.g. MLX) is thread-safe for its own
// operations.
//
// Design rationale: all methods take Storage by const-reference and return
// a new Storage.  No in-place mutation is exposed at this interface level.
// Concrete implementations (CpuBackend / GpuBackend) choose the most
// efficient representation internally.
class IBackend {
public:
    virtual ~IBackend() = default;

    // Returns the Device tag associated with this backend.
    virtual Device device() const noexcept = 0;

    // Transfers a CPU buffer into the backend's native storage format.
    // For CpuBackend this is a no-op move; for GpuBackend it copies data
    // to GPU-private memory via mlx::core::copy().
    virtual Storage from_cpu(CpuStorage cpu, const Shape& shape) = 0;

    // Allocates a zero-filled buffer of shape `shape` with element type `dt`.
    virtual Storage zeros(const Shape& shape, Dtype dt) = 0;

    // Allocates a one-filled buffer of shape `shape` with element type `dt`.
    virtual Storage ones(const Shape& shape, Dtype dt) = 0;

    // Returns a deep copy of `src` reinterpreted with the given shape/dtype.
    virtual Storage clone(const Storage& src, const Shape& shape, Dtype dt) = 0;

    // Returns a contiguous copy of a (possibly strided or offset) view.
    //
    // When already_contiguous is true and storage_offset is 0 the
    // implementation may use a fast memcpy path.  Otherwise it must walk the
    // stride/offset layout to produce a densely-packed output buffer.
    virtual Storage contiguous(const Storage& src,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset,
                               bool already_contiguous,
                               Dtype dt) = 0;

    // Elementwise arithmetic: a op b, both already broadcast to `shape`.
    virtual Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Elementwise bitwise operation selected by op: 0=AND, 1=OR, 2=XOR.
    // Only valid for integer and Bool dtypes.
    virtual Storage
    bitwise_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;

    // Elementwise comparison returning a Bool Storage of the same shape.
    // op: 0=EQ, 1=NE, 2=GT, 3=GE, 4=LT, 5=LE.
    virtual Storage
    compare_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;

    // Elementwise max(a, b) and min(a, b).
    virtual Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Elementwise unary math ops.  All operate element-by-element on the
    // flattened buffer; shape is used only for numel computation.
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

    // Additional elementwise unary ops: logarithmic, trigonometric, and
    // activation functions not covered by the basic set above.
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

    // Bitwise NOT (integer types only).
    virtual Storage invert(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Activation functions that need both a forward and a backward pass.
    // The *_backward variants receive the pre-activation input `a` and the
    // upstream gradient `grad`, and return the input gradient.
    virtual Storage silu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage gelu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    gelu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) = 0;
    virtual Storage softplus(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) = 0;
    virtual Storage elu_backward(
        const Storage& a, const Storage& grad, const Shape& shape, Dtype dt, double alpha) = 0;
    virtual Storage selu(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    selu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage mish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    mish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    hard_sigmoid_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;
    virtual Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    hard_swish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) = 0;

    // ReLU clamped to [0, 6], used in MobileNet-style networks.
    virtual Storage relu6(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Numerically stable log-softmax along a single axis.
    virtual Storage log_softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Backward pass for log_softmax:
    //   dL/dx = dL/dy - exp(y) * sum(dL/dy, axis, keepdims=true)
    // where y = log_softmax(x) is the saved forward output.
    virtual Storage log_softmax_backward(
        const Storage& y, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) = 0;

    // Boolean reductions — reduce the full tensor to a scalar bool.
    virtual Storage any(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage all(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Floating-point predicate ops (output dtype is always Bool).
    virtual Storage isinf(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage isnan(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage isfinite(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Replace NaN/Inf values with finite substitutes.  Output dtype matches input.
    virtual Storage nan_to_num(
        const Storage& a, const Shape& shape, Dtype dt,
        double nan_val, double posinf_val, double neginf_val) = 0;

    // Axis reductions.  opts.axes specifies which dimensions to collapse; the
    // output shape follows keepdims semantics.  variance() uses the biased
    // (population) variance formula to match NumPy/reference defaults.
    virtual Storage
    reduce_sum(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    reduce_mean(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    variance(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;
    virtual Storage
    reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) = 0;

    // Scan operations along a single axis.
    virtual Storage cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Stable numerically-safe softmax forward and its vector-Jacobian product.
    // softmax_backward receives the softmax output `z` (not the pre-activation
    // input) because the gradient formula only needs the output: dx = z*(dz - (z·dz)).
    virtual Storage softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage softmax_backward(
        const Storage& z, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) = 0;

    // Reverses elements along `axis` (equivalent to a[::-1] on that dimension).
    virtual Storage
    reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Sum of the main diagonal elements; trace_backward scatters the upstream
    // gradient back to the diagonal positions of the input shape.
    virtual Storage trace(const Storage& a, const Shape& shape, Dtype dt) = 0;
    virtual Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) = 0;

    // Constructs N-D coordinate grids from 1-D input vectors.
    // indexing_xy=true uses x-column / y-row convention (MATLAB-style).
    virtual std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                          const Shape& out_shape,
                                          Dtype dt,
                                          bool indexing_xy) = 0;

    // Selects grad where cond is true (true_branch=true) or false, zeroing
    // the other branch.  Used by the Where autograd backward.
    virtual Storage where_branch(const Storage& grad,
                                 const Storage& cond,
                                 const Shape& shape,
                                 Dtype dt,
                                 bool true_branch) = 0;

    // Replaces elements of `a` where `mask` is nonzero with `value`.
    virtual Storage masked_fill(
        const Storage& a, const Storage& mask, const Shape& shape, Dtype dt, double value) = 0;
    // Gathers slices from `a` along `axis` using integer `indices`.
    // index_dtype is the dtype of the indices tensor (I32 or I64).
    virtual Storage gather(const Storage& a,
                           const Storage& indices,
                           const Shape& input_shape,
                           const Shape& output_shape,
                           int axis,
                           Dtype index_dtype,
                           Dtype dt) = 0;

    // Backward pass for gather: scatters grad values back to the source
    // positions identified by indices.  Result has input_shape.
    virtual Storage gather_backward(const Storage& grad,
                                    const Storage& indices,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    Dtype index_dtype,
                                    Dtype dt) = 0;

    // Extracts a generalized diagonal between axis1 and axis2 with an
    // integer offset from the main diagonal (positive = above, negative = below).
    virtual Storage diagonal(
        const Storage& a, const Shape& input_shape, int offset, int axis1, int axis2, Dtype dt) = 0;

    // Backward pass for diagonal: scatters the diagonal gradient back into a
    // zero-filled tensor of input_shape.
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
    virtual Storage
    reshape(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) = 0;
    virtual Storage slice_axis(const Storage& a,
                               const Shape& src_shape,
                               const Shape& slice_shape,
                               int axis,
                               std::int64_t offset,
                               Dtype dt) = 0;
    // Inverse of slice_axis: scatters `a` into a zero-filled tensor of
    // dst_shape at the given offset along axis.  Used in slice backward passes.
    virtual Storage insert_axis_slice(const Storage& a,
                                      const Shape& src_shape,
                                      const Shape& dst_shape,
                                      int axis,
                                      std::int64_t offset,
                                      Dtype dt) = 0;

    // Concatenates the tensors in `xs` along `axis`; each shape in `shapes`
    // corresponds element-wise to `xs`.
    virtual Storage concatenate(const std::vector<Storage>& xs,
                                const std::vector<Shape>& shapes,
                                int axis,
                                Dtype dt) = 0;

    // Stacks tensors along a new axis inserted at position `axis`.
    // All inputs must have the same `input_shape`.
    virtual Storage
    stack(const std::vector<Storage>& xs, const Shape& input_shape, int axis, Dtype dt) = 0;

    // Splits `a` into num_splits equal-sized pieces along `axis`.
    virtual std::vector<Storage> split_equal(
        const Storage& a, const Shape& shape, int axis, std::int64_t num_splits, Dtype dt) = 0;

    // Splits `a` at the given index boundaries along `axis` (like np.split).
    virtual std::vector<Storage> split_at(const Storage& a,
                                          const Shape& shape,
                                          int axis,
                                          const std::vector<std::int64_t>& indices,
                                          Dtype dt) = 0;

    // Backward pass for repeat(): reduces repeated slices by summing them back
    // to the original input_shape.
    virtual Storage repeat_backward(const Storage& grad_out,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    int axis,
                                    std::int64_t repeats,
                                    Dtype dt) = 0;

    // Backward pass for tile(): accumulates gradients from all tiled copies
    // back into the original input_shape.
    virtual Storage tile_backward(const Storage& grad_out,
                                  const Shape& input_shape,
                                  const Shape& padded_shape,
                                  const Shape& output_shape,
                                  const std::vector<std::int64_t>& reps,
                                  Dtype dt) = 0;

    // Returns (sorted_values, sorted_indices) along `axis`.
    // descending=true sorts from largest to smallest.
    virtual std::pair<Storage, Storage> sort_select(const Storage& a,
                                                    const Shape& input_shape,
                                                    const Shape& output_shape,
                                                    int axis,
                                                    Dtype dt,
                                                    bool descending) = 0;

    // Returns the permutation indices that would sort `a` along `axis`.
    virtual Storage argsort(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;

    // Returns the index of the max (is_min=false) or min (is_min=true) along `axis`.
    // Output dtype is I64.
    virtual Storage arg_reduce_index(
        const Storage& a, const Shape& shape, int axis, bool keepdims, Dtype dt, bool is_min) = 0;

    // Accumulates `grad` values into output_shape positions specified by `indices`
    // along `axis` (sparse scatter-add / gather backward).
    virtual Storage scatter_add_axis(const Storage& grad,
                                     const Storage& indices,
                                     const Shape& output_shape,
                                     const Shape& grad_shape,
                                     int axis,
                                     Dtype dt) = 0;

    // User-facing scatter-add: out[..., index[i], ...] += src[..., i, ...]
    // along dim. base is unchanged except at index positions where src is added.
    virtual Storage scatter_add(const Storage& base,
                                const Storage& indices,
                                const Storage& src,
                                const Shape& base_shape,
                                const Shape& idx_shape,
                                int dim,
                                Dtype dt) = 0;

    // Sliding-window view along a single dimension.
    // Returns shape (*base.shape[:dim], L, *base.shape[dim+1:], size)
    // where L = (dim_size - size) / step + 1.
    virtual Storage unfold_dim(const Storage& a,
                               const Shape& in_shape,
                               int dim,
                               int size,
                               int step,
                               Dtype dt) = 0;

    // General batched matrix multiplication.  Shapes and transpose flags are
    // encoded in opts; the implementation must handle both 2-D and batched cases.
    virtual Storage
    matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) = 0;

    // Fused linear projection: out = x @ weight.T + bias.
    // x_shape is (*, K), weight_shape is (N, K), out_shape is (*, N).
    virtual Storage linear(const Storage& x,
                           const Storage& weight,
                           const Storage& bias,
                           const Shape& x_shape,
                           const Shape& weight_shape,
                           const Shape& out_shape,
                           Dtype dt) = 0;

    // Backward pass for linear; returns [grad_x, grad_weight, grad_bias].
    virtual std::vector<Storage> linear_backward(const Storage& grad,
                                                 const Storage& x,
                                                 const Storage& weight,
                                                 const Shape& x_shape,
                                                 const Shape& weight_shape,
                                                 const Shape& bias_shape,
                                                 Dtype dt) = 0;
    // RMSNorm forward pass.  outer = product of batch/sequence dims; normalized_size
    // is the last (feature) dimension.  Returns (output, saved_rstd); saved_rstd
    // holds per-row reciprocal standard deviations needed by the backward pass.
    virtual StoragePair rms_norm_forward(const Storage& x,
                                         const Storage& gamma,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         double eps,
                                         const Shape& x_shape,
                                         Dtype dt) = 0;

    // RMSNorm backward; returns (grad_x, grad_gamma).
    virtual StoragePair rms_norm_backward(const Storage& x,
                                          const Storage& gamma,
                                          const Storage& saved_rstd,
                                          const Storage& grad,
                                          std::size_t outer,
                                          std::size_t normalized_size,
                                          const Shape& x_shape,
                                          const Shape& gamma_shape,
                                          Dtype dt) = 0;

    // LayerNorm forward.  Returns [output, saved_mean, saved_rstd].
    virtual std::vector<Storage> layer_norm_forward(const Storage& x,
                                                    const Storage& gamma,
                                                    const Storage& beta,
                                                    std::size_t outer,
                                                    std::size_t normalized_size,
                                                    double eps,
                                                    const Shape& x_shape,
                                                    Dtype dt) = 0;

    // LayerNorm backward; returns [grad_x, grad_gamma, grad_beta].
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
    // BatchNorm training forward.  ndim distinguishes 1-D, 2-D, and 3-D spatial
    // tensors.  Returns [output, saved_mean, saved_rstd].
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

    // BatchNorm backward; returns [grad_x, grad_gamma, grad_beta].
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

    // GroupNorm forward; groups partitions channels into equal groups before
    // normalizing each group independently.  Returns [output, saved_mean, saved_rstd].
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

    // GroupNorm backward; returns [grad_x, grad_gamma, grad_beta].
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
    // Linear algebra ops.  All accept batched inputs where the last two dims
    // form the matrix.  CPU implementations delegate to LAPACK (via Accelerate);
    // GPU implementations delegate to MLX linalg.

    // Vector/matrix/Frobenius norm; ord controls the norm type.
    virtual Storage linalg_norm(const Storage& a,
                                const Shape& shape,
                                double ord,
                                const std::vector<int>& axes,
                                bool keepdims,
                                Dtype dt) = 0;

    // Cholesky decomposition.  upper=true returns the upper triangular factor.
    virtual Storage linalg_cholesky(const Storage& a, const Shape& shape, bool upper, Dtype dt) = 0;

    // Matrix inverse using LU factorisation.
    virtual Storage linalg_inv(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Solves a linear system Ax = b; b may have multiple right-hand sides.
    virtual Storage linalg_solve(const Storage& a,
                                 const Storage& b,
                                 const Shape& a_shape,
                                 const Shape& b_shape,
                                 Dtype dt) = 0;

    // A^power computed via repeated squaring or eigendecomposition.
    virtual Storage
    linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) = 0;

    // Moore-Penrose pseudoinverse via SVD.
    virtual Storage linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Determinant (scalar per batch element).
    virtual Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Reduced QR decomposition; returns (Q, R).
    virtual StoragePair linalg_qr(const Storage& a,
                                  const Shape& shape,
                                  const Shape& q_shape,
                                  const Shape& r_shape,
                                  Dtype dt) = 0;

    // General eigendecomposition for non-symmetric matrices; returns (eigenvalues, eigenvectors).
    // Eigenvalues may be complex; CPU backend returns packed real+imag components.
    virtual StoragePair linalg_eig(const Storage& a,
                                   const Shape& shape,
                                   const Shape& values_shape,
                                   const Shape& vectors_shape,
                                   Dtype dt) = 0;

    // Symmetric/Hermitian eigendecomposition (real eigenvalues guaranteed).
    // Returns (eigenvalues, eigenvectors); eigenvectors are column-ordered.
    virtual StoragePair linalg_eigh(const Storage& a,
                                    const Shape& shape,
                                    const Shape& values_shape,
                                    const Shape& vectors_shape,
                                    Dtype dt) = 0;

    // Singular value decomposition.  compute_uv=false returns only singular
    // values S.  Returns [U, S, Vt] or [S] depending on compute_uv.
    virtual std::vector<Storage> linalg_svd(const Storage& a,
                                            const Shape& shape,
                                            bool compute_uv,
                                            const Shape& u_shape,
                                            const Shape& s_shape,
                                            const Shape& vt_shape,
                                            Dtype dt) = 0;

    // LU factorisation (packed format).  Returns {LU, pivots} where LU is
    // the packed n×n matrix (LAPACK dgetrf_ format) and pivots is an n-element
    // I32 tensor of 1-based pivot indices.
    virtual StoragePair linalg_lu_factor(const Storage& a,
                                         const Shape& shape,
                                         Dtype dt) = 0;

    // Triangular solve: solve A X = B where A is triangular.
    // upper=true → upper triangular; unit=true → unit diagonal.
    virtual Storage linalg_solve_triangular(const Storage& a,
                                             const Storage& b,
                                             const Shape& a_shape,
                                             const Shape& b_shape,
                                             bool upper,
                                             bool unitriangular,
                                             Dtype dt) = 0;

    // Least-squares: solve min||AX-B||_2. Returns [solution, residuals, rank, svd].
    // a_shape=(m,n), b_shape=(m,nrhs). Solution shape=(n,nrhs).
    virtual std::vector<Storage> linalg_lstsq(const Storage& a,
                                               const Storage& b,
                                               const Shape& a_shape,
                                               const Shape& b_shape,
                                               Dtype dt) = 0;

    // Solve AX=B given LU+pivots from linalg_lu_factor. Returns X (same shape as B).
    virtual Storage linalg_lu_solve(const Storage& LU,
                                     const Storage& pivots,
                                     const Storage& b,
                                     const Shape& lu_shape,
                                     const Shape& b_shape,
                                     Dtype dt) = 0;

    // Reconstruct Q (m×k) from Householder reflectors H (m×n) and tau (k,).
    virtual Storage linalg_householder_product(const Storage& H,
                                                const Storage& tau,
                                                const Shape& h_shape,
                                                Dtype dt) = 0;

    // LDL^T factorization of symmetric matrix. Returns {LD_packed, pivots_i32}.
    virtual StoragePair linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Broadcasts `a` from src_shape to dst_shape following NumPy rules.
    virtual Storage
    broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) = 0;

    // Repeats elements `repeats` times along `axis` (equivalent to np.repeat).
    virtual Storage
    repeat(const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) = 0;

    // Tiles the tensor by replicating it reps[i] times along each dimension.
    virtual Storage
    tile(const Storage& a, const Shape& shape, Dtype dt, const std::vector<std::int64_t>& reps) = 0;

    // Permutes dimensions according to `perm` (equivalent to np.transpose).
    virtual Storage
    permute(const Storage& a, const Shape& shape, const std::vector<int>& perm, Dtype dt) = 0;

    // Zero-pads (or reflects/wraps) the tensor with pad_width[d] = (before, after)
    // extra elements for each dimension d.  constant specifies the fill value.
    virtual Storage pad(const Storage& a,
                        const Shape& shape,
                        Dtype dt,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        double constant) = 0;

    // Raises each element to a scalar exponent: a**exp.
    virtual Storage pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) = 0;

    // Raises a scalar base to the power of each element: base**a.
    virtual Storage rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) = 0;

    // Clamps each element to [min_v, max_v].
    virtual Storage
    clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) = 0;

    // Type-casts the buffer element-by-element from src_dt to dst_dt.
    virtual Storage cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) = 0;

    // Loss functions.  reduction: 0=elementwise (no reduction), 1=mean, 2=sum.
    // Backward passes return one gradient tensor per differentiable input.

    // Mean-squared error: 0.5*(input - target)^2 summed/averaged.
    // Backward returns (grad_input, grad_target).
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

    // Huber loss (smooth L1): quadratic for |r|<=delta, linear otherwise.
    // delta controls the transition point between the two regimes.
    // Backward returns (grad_input, grad_target).
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

    // Cross-entropy loss with optional per-class weights and ignore_index.
    // saved_aux inside the result holds the log-softmax activations needed by
    // the backward pass.  weight is optional (may be nullptr for unweighted CE).
    virtual ClassLossForwardResult cross_entropy_loss(const Storage& input,
                                                      const Storage& target,
                                                      const Storage* weight,
                                                      const Shape& input_shape,
                                                      const Shape& target_shape,
                                                      Dtype dt,
                                                      double eps,
                                                      int ignore_index,
                                                      int reduction) = 0;

    // Cross-entropy backward; saved_softmax is the softmax output stored during
    // the forward.  valid_count accounts for ignored-index samples in mean reduction.
    virtual Storage cross_entropy_backward(const Storage& saved_softmax,
                                           const Storage& target,
                                           const Storage* weight,
                                           const Storage& valid_count,
                                           const Storage& grad,
                                           const Shape& input_shape,
                                           Dtype dt,
                                           int ignore_index,
                                           int reduction) = 0;

    // Negative log-likelihood loss; operates on log-probabilities directly
    // (unlike cross_entropy which applies log-softmax internally).
    virtual ClassLossForwardResult nll_loss(const Storage& input,
                                            const Storage& target,
                                            const Storage* weight,
                                            const Shape& input_shape,
                                            const Shape& target_shape,
                                            Dtype dt,
                                            int ignore_index,
                                            int reduction) = 0;

    // NLL backward; returns the input gradient.
    virtual Storage nll_loss_backward(const Storage& target,
                                      const Storage* weight,
                                      const Storage& valid_count,
                                      const Storage& grad,
                                      const Shape& input_shape,
                                      Dtype dt,
                                      int ignore_index,
                                      int reduction) = 0;

    // Materialises the tensor's data in CPU-accessible memory.  For GpuBackend
    // this calls mlx::core::array::eval() and copies to a fresh CpuStorage.
    virtual CpuStorage to_cpu(const Storage& a, const Shape& shape) = 0;

    // Scaled dot-product attention forward.  Returns [output, saved_weights]
    // where saved_weights holds the post-softmax attention scores for backward.
    // is_causal=true applies a lower-triangular mask to prevent attending to
    // future tokens without materialising the full mask tensor.
    virtual std::vector<Storage> sdpa_forward(const Storage& q,
                                              const Storage& k,
                                              const Storage& v,
                                              const Storage* attn_mask,
                                              const Shape& q_shape,
                                              const Shape& k_shape,
                                              const Shape& v_shape,
                                              Dtype mask_dtype,
                                              std::size_t mask_numel,
                                              double scale,
                                              bool is_causal,
                                              Dtype dt) = 0;

    // SDPA backward; returns [grad_q, grad_k, grad_v].
    virtual std::vector<Storage> sdpa_backward(const Storage& grad_out,
                                               const Storage& q,
                                               const Storage& k,
                                               const Storage& v,
                                               const Storage& saved_weights,
                                               const Shape& q_shape,
                                               const Shape& k_shape,
                                               const Shape& v_shape,
                                               double scale,
                                               Dtype dt) = 0;

    virtual Storage conv_transpose_nd_forward(const Storage& x,
                                              const Storage& W,
                                              const Storage& b,
                                              int B,
                                              int Cin,
                                              int Cout,
                                              const int* S,
                                              const int* K,
                                              const int* O,
                                              const int* stride,
                                              const int* pad,
                                              const int* opad,
                                              int N,
                                              const Shape& out_shape,
                                              Dtype dt) = 0;

    virtual std::vector<Storage> conv_transpose_nd_backward(const Storage& grad_out,
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

    // Compact parameter bag for N-D standard convolution (N in {1,2,3}).
    // Arrays are indexed 0..N-1; unused trailing entries are ignored.
    struct ConvNdOpts {
        int N;          // number of spatial dimensions
        int groups;     // channel grouping factor for grouped convolution
        int stride[3];
        int pad[3];
        int dilation[3];
    };

    virtual Storage conv_nd_forward(const Storage& x,
                                    const Storage& W,
                                    const Storage& b,
                                    int B,
                                    int Cin,
                                    int Cout,
                                    int Cin_g,
                                    int Cout_g,
                                    const int* S,
                                    const int* K,
                                    const int* O,
                                    const ConvNdOpts& opts,
                                    const Shape& out_shape,
                                    Dtype dt) = 0;

    virtual std::vector<Storage> conv_nd_backward(const Storage& grad_out,
                                                  const Storage& x,
                                                  const Storage& W,
                                                  int B,
                                                  int Cin,
                                                  int Cout,
                                                  int Cin_g,
                                                  int Cout_g,
                                                  const int* S,
                                                  const int* K,
                                                  const int* O,
                                                  const ConvNdOpts& opts,
                                                  Dtype dt) = 0;

    virtual Storage unfold_forward(const Storage& x,
                                   int B,
                                   int C,
                                   const std::vector<int>& S,
                                   const std::vector<int>& K,
                                   const std::vector<int>& O,
                                   const std::vector<int>& stride,
                                   const std::vector<int>& pad,
                                   const std::vector<int>& dilation,
                                   const Shape& out_shape,
                                   Dtype dt) = 0;

    virtual Storage unfold_backward(const Storage& grad_out,
                                    int B,
                                    int C,
                                    const std::vector<int>& S,
                                    const std::vector<int>& K,
                                    const std::vector<int>& O,
                                    const std::vector<int>& stride,
                                    const std::vector<int>& pad,
                                    const std::vector<int>& dilation,
                                    Dtype dt) = 0;

    virtual std::pair<Storage, Storage> expand_and_multiply(const Storage& mask,
                                                            const Storage& x,
                                                            const Shape& mask_shape,
                                                            const Shape& x_shape,
                                                            Dtype dt) = 0;

    virtual Storage drop_block_mask(const Storage& seed,
                                    double drop_prob,
                                    std::int64_t block_size,
                                    const Shape& x_shape,
                                    Dtype dt) = 0;

    // Embedding table lookup: out[i] = weight[indices[i]].
    // padding_idx entries are zeroed in the output.
    virtual Storage embedding_forward(const Storage& weight,
                                      const Storage& indices,
                                      const Shape& weight_shape,
                                      const Shape& indices_shape,
                                      const Shape& out_shape,
                                      int padding_idx,
                                      Dtype dt) = 0;

    // Backward pass for embedding: scatter-adds grad_out rows into a
    // zero-filled weight gradient of weight_shape.
    virtual Storage embedding_backward(const Storage& grad_out,
                                       const Storage& indices,
                                       const Shape& weight_shape,
                                       const Shape& indices_shape,
                                       int padding_idx,
                                       Dtype dt) = 0;

    // Fold (col2im): inverse of unfold/im2col. Accumulates (N, C*kH*kW, L)
    // patches back into (N, C, outH, outW) using scatter-add.
    virtual Storage nn_fold(const Storage& x,
                             const Shape& x_shape,
                             const Shape& out_shape,
                             const std::vector<int>& kernel_size,
                             const std::vector<int>& stride,
                             const std::vector<int>& padding,
                             const std::vector<int>& dilation,
                             Dtype dt) = 0;

    // EmbeddingBag: gather rows from weight at indices, then reduce per bag.
    // mode: 0=sum, 1=mean, 2=max. For 1-D indices, offsets marks bag starts.
    virtual Storage embedding_bag_forward(const Storage& weight,
                                           const Storage& indices,
                                           const Storage& offsets,
                                           const Shape& weight_shape,
                                           const Shape& indices_shape,
                                           int mode,
                                           int padding_idx,
                                           bool include_last_offset,
                                           Dtype dt) = 0;

    // Flip (reverse) along the given axes.
    virtual Storage flip(const Storage& a, const Shape& shape,
                         const std::vector<int>& dims, Dtype dt) = 0;

    // Masked select: extract elements where bool mask == true.
    // Returns a flat 1-D Storage of `n_true` elements.
    virtual Storage masked_select_count(const Storage& mask,
                                        const Shape& shape, Dtype dt) = 0;
    virtual Storage masked_select(const Storage& a, const Storage& mask,
                                   const Shape& a_shape, const Shape& mask_shape,
                                   std::int64_t n_true, Dtype dt) = 0;

    // CTC (Connectionist Temporal Classification) loss.
    // log_probs: (T, N, C) log-probabilities; targets: (N, S) or flat (sum_S,)
    // input_lengths: (N,); target_lengths: (N,).
    // Returns per-sample losses of shape (N,) before reduction.
    virtual Storage ctc_loss_forward(const Storage& log_probs,
                                     const Storage& targets,
                                     const Storage& input_lengths,
                                     const Storage& target_lengths,
                                     const Shape& lp_shape,
                                     int blank,
                                     bool zero_infinity,
                                     Dtype dt) = 0;

    // Generates a fixed sinusoidal positional encoding of shape
    // (seq_len, embed_dim) following the original Transformer paper formula.
    virtual Storage
    sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) = 0;

    // Rotary Position Embedding (RoPE) forward.  Returns [output, cos_cache, sin_cache].
    // interleaved=true uses GPT-NeoX layout; false uses Llama layout.
    // position_ids is optional; if null, positions 0..seq_len-1 are assumed.
    virtual std::vector<Storage> rope_forward(const Storage& x,
                                              const Storage* position_ids,
                                              const Shape& x_shape,
                                              bool interleaved,
                                              Dtype pos_dtype,
                                              Dtype dt) = 0;

    // RoPE backward; applies the inverse rotation using cached cos/sin values.
    virtual Storage rope_backward(const Storage& grad_out,
                                  const Storage& saved_cos,
                                  const Storage& saved_sin,
                                  const Shape& x_shape,
                                  bool interleaved,
                                  Dtype dt) = 0;

    // Compact parameter bag for N-D pooling (N in {1,2,3}).
    struct PoolOpts {
        int K[3];       // kernel size per spatial dimension
        int stride[3];  // stride per spatial dimension
        int pad[3];     // symmetric padding per spatial dimension
        int N;          // number of spatial dimensions
    };

    // N-D max pooling forward.  Returns [output, argmax]; argmax stores the
    // flat index within the spatial window that produced each maximum, required
    // by max_pool_nd_backward to route gradients.
    virtual std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                                     const Shape& x_shape,
                                                     const Shape& out_shape,
                                                     const PoolOpts& opts,
                                                     Dtype dt) = 0;

    // N-D max pooling backward; scatters grad_out to the argmax positions.
    virtual Storage max_pool_nd_backward(const Storage& grad_out,
                                         const Storage& saved_argmax,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    // N-D average pooling forward (count-include-pad semantics by default).
    virtual Storage avg_pool_nd_forward(const Storage& x,
                                        const Shape& x_shape,
                                        const Shape& out_shape,
                                        const PoolOpts& opts,
                                        Dtype dt) = 0;

    // N-D average pooling backward; distributes grad uniformly across each window.
    virtual Storage avg_pool_nd_backward(const Storage& grad_out,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    // BatchNorm inference-mode forward; uses running statistics (mean, var)
    // rather than computing batch statistics.  Returns [output, rstd].
    virtual std::vector<Storage> batch_norm_eval_forward(const Storage& x,
                                                         const Storage& mean,
                                                         const Storage& var,
                                                         const Storage& gamma,
                                                         const Storage& beta,
                                                         const Shape& x_shape,
                                                         int C,
                                                         int spatial,
                                                         double eps,
                                                         Dtype dt) = 0;

    // BatchNorm inference-mode backward; returns [grad_x, grad_gamma, grad_beta].
    virtual std::vector<Storage> batch_norm_eval_backward(const Storage& x,
                                                          const Storage& mean,
                                                          const Storage& gamma,
                                                          const Storage& rstd,
                                                          const Storage& grad_out,
                                                          const Shape& x_shape,
                                                          int C,
                                                          int spatial,
                                                          Dtype dt) = 0;

    // Lp-normalizes `x` along `axis` using the given norm order `ord`.
    // Returns [output, saved_norm] so the backward can reuse the computed norm.
    virtual std::vector<Storage> lp_normalize_forward(
        const Storage& x, const Shape& x_shape, double ord, int axis, double eps, Dtype dt) = 0;

    // Lp-normalize backward; saved_norm is the per-row norm from the forward.
    virtual Storage lp_normalize_backward(const Storage& x,
                                          const Storage& saved_norm,
                                          const Storage& grad_out,
                                          const Shape& x_shape,
                                          double ord,
                                          int axis,
                                          Dtype dt) = 0;

    // Global Response Normalization (GRN) from ConvNeXt-v2.
    // Returns [output, saved_Nx] where Nx is the per-channel L2 norm.
    virtual std::vector<Storage> global_response_norm_forward(const Storage& x,
                                                              const Storage& gamma,
                                                              const Storage& beta,
                                                              const Shape& x_shape,
                                                              double eps,
                                                              Dtype dt) = 0;

    // GRN backward; returns [grad_x, grad_gamma, grad_beta].
    virtual std::vector<Storage> global_response_norm_backward(const Storage& x,
                                                               const Storage& gamma,
                                                               const Storage& beta,
                                                               const Storage& saved_Nx,
                                                               const Storage& grad_out,
                                                               const Shape& x_shape,
                                                               double eps,
                                                               Dtype dt) = 0;

    // Nearest-neighbor upsampling for 2-D and 3-D feature maps.
    virtual Storage interpolate_nearest_2d_forward(
        const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) = 0;

    virtual Storage interpolate_nearest_3d_forward(
        const Storage& input, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) = 0;

    // Bilinear interpolation forward/backward.
    // align_corners=true maps corner pixels to corner coordinates; false
    // scales by factor (in_size / out_size).
    virtual Storage interpolate_bilinear_forward(const Storage& input,
                                                 const Shape& in_shape,
                                                 int H_out,
                                                 int W_out,
                                                 bool align_corners,
                                                 Dtype dt) = 0;

    virtual Storage interpolate_bilinear_backward(const Storage& grad_out,
                                                  const Shape& in_shape,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) = 0;

    // Trilinear interpolation for volumetric feature maps.
    virtual Storage interpolate_trilinear_forward(const Storage& input,
                                                  const Shape& in_shape,
                                                  int D_out,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) = 0;

    virtual Storage interpolate_trilinear_backward(const Storage& grad_out,
                                                   const Shape& in_shape,
                                                   int D_out,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners,
                                                   Dtype dt) = 0;

    // Affine grid generation for spatial transformer networks.
    // theta is an (N, 2, 3) affine matrix; output has shape (N, H, W, 2).
    virtual Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) = 0;

    // Affine grid backward: returns grad_theta of shape (N, 2, 3).
    virtual Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) = 0;

    // Samples input at grid locations using bilinear (mode=0) or nearest (mode=1)
    // interpolation.  padding_mode: 0=zeros, 1=border, 2=reflection.
    virtual Storage grid_sample_forward(const Storage& input,
                                        const Storage& grid,
                                        const Shape& in_shape,
                                        const Shape& grid_shape,
                                        int mode,
                                        int padding_mode,
                                        bool align_corners,
                                        Dtype dt) = 0;

    // Grid sample backward; returns [grad_input, grad_grid].
    virtual std::vector<Storage> grid_sample_backward(const Storage& grad_out,
                                                      const Storage& input,
                                                      const Storage& grid,
                                                      const Shape& in_shape,
                                                      const Shape& grid_shape,
                                                      int mode,
                                                      int padding_mode,
                                                      bool align_corners,
                                                      Dtype dt) = 0;

    virtual Storage bilinear_layer_forward(const Storage& x1,
                                           const Storage& x2,
                                           const Storage& weight,
                                           const Storage& bias,
                                           bool has_bias,
                                           const Shape& x1_shape,
                                           const Shape& x2_shape,
                                           const Shape& w_shape,
                                           Dtype dt) = 0;

    virtual std::vector<Storage> bilinear_layer_backward(const Storage& grad_out,
                                                         const Storage& x1,
                                                         const Storage& x2,
                                                         const Storage& weight,
                                                         const Shape& x1_shape,
                                                         const Shape& x2_shape,
                                                         const Shape& w_shape,
                                                         bool has_bias,
                                                         Dtype dt) = 0;

    // One-hot encoding: produces a (num_classes,)-wide indicator tensor for
    // each index in `indices`.  Output dtype is typically F32 or I32.
    virtual Storage one_hot_forward(const Storage& indices,
                                    const Shape& indices_shape,
                                    int num_classes,
                                    Dtype out_dtype) = 0;

    // Rotates a 2-D image by angle_rad_neg (stored negated for efficiency)
    // around center (cx, cy) using bilinear interpolation.
    virtual Storage rotate_forward(const Storage& input,
                                   const Shape& shape,
                                   double angle_rad_neg,
                                   double cx,
                                   double cy,
                                   Dtype dt) = 0;

    // Element-wise masks that return 0.0/1.0 float results.
    // ge_mask: out[i] = (a[i] >= b[i]) ? 1 : 0
    // lt_mask: out[i] = (a[i] < b[i])  ? 1 : 0
    virtual Storage ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Broadcasts a scalar to the array and adds/multiplies element-wise.
    virtual Storage add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) = 0;
    virtual Storage mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) = 0;

    // Returns a 0/1 mask where a[i] is in the closed interval [lo, hi].
    virtual Storage
    in_range_mask(const Storage& a, const Shape& shape, Dtype dt, double lo, double hi) = 0;

    // Returns slope*a[i] when a[i]<0, else a[i].  Used inside leaky-relu backward.
    virtual Storage leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) = 0;

    // Returns 1.0 where a[i] > 0, else 0.0.  Used in relu backward.
    virtual Storage positive_mask(const Storage& a, const Shape& shape, Dtype dt) = 0;

    // Reduces `grad` (of shape grad_shape) to target_shape by summing over the
    // broadcast dimensions.  Used in broadcast backward passes.
    virtual Storage reduce_grad_to_shape(const Storage& grad,
                                         const Shape& grad_shape,
                                         const Shape& target_shape,
                                         Dtype dt) = 0;

    // Broadcasts `grad` back to input_shape after a reduction.  The inverse of
    // reduce_sum when keepdims=false.
    virtual Storage broadcast_back_for_reduce(const Storage& grad,
                                              const Shape& grad_shape,
                                              const Shape& input_shape,
                                              const std::vector<int>& axes,
                                              bool keepdims,
                                              Dtype dt) = 0;

    // Allocates a tensor filled with fill_value.
    virtual Storage full(const Shape& shape, Dtype dt, double fill_value) = 0;

    // Constructs an M×N identity-like matrix with the diagonal at offset k.
    virtual Storage eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) = 0;

    // Extracts a 1-D diagonal at offset k from a 2-D matrix, or places a 1-D
    // vector on a 2-D diagonal.  out_shape is written with the result shape.
    virtual Storage
    diag(const Storage& v, const Shape& v_shape, std::int64_t k, Dtype dt, Shape& out_shape) = 0;

    // Applies a triangular mask: upper=true zeroes the lower triangle,
    // upper=false zeroes the upper triangle, relative to diagonal offset k.
    virtual Storage tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) = 0;

    // Integer floor division: out[i] = floor(a[i] / b[i]).
    virtual Storage floordiv(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    // Inner product generalised to batches: contracts the last axis of `a`
    // against the last axis of `b`.
    virtual Storage inner(const Storage& a,
                          const Storage& b,
                          const Shape& a_shape,
                          const Shape& b_shape,
                          const Shape& out_shape,
                          Dtype dt) = 0;

    // Permutes a CpuStorage buffer in-place and returns the result as a new
    // CpuStorage.  Used for data layout conversions before GPU upload.
    virtual CpuStorage permute_cpu(const CpuStorage& src,
                                   const Shape& src_shape,
                                   const std::vector<int>& perm,
                                   Dtype dt) = 0;

    // General tensor contraction (generalised matmul) along axes_a and axes_b.
    virtual Storage tensordot(const Storage& a,
                              const Storage& b,
                              const Shape& a_shape,
                              const Shape& b_shape,
                              const Shape& out_shape,
                              const std::vector<int>& axes_a,
                              const std::vector<int>& axes_b,
                              Dtype dt) = 0;

    // Element-wise selection: out[i] = cond[i] ? x[i] : y[i].
    virtual Storage where_op(
        const Storage& cond, const Storage& x, const Storage& y, const Shape& shape, Dtype dt) = 0;

    // Sums grad over the broadcast dimensions so that the result has
    // output_shape (the pre-broadcast shape).
    virtual Storage reduce_broadcast(const Storage& grad,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     Dtype dt) = 0;

    // Computes a histogram over `input` with `bins` equal-width buckets in
    // [lo, hi].  density=true normalises the result to a probability density.
    virtual Storage histogram_forward(const Storage& input,
                                      const Shape& input_shape,
                                      Dtype input_dtype,
                                      double lo,
                                      double hi,
                                      std::int64_t bins,
                                      bool density) = 0;

    // Returns a CpuStorage containing the flat indices of all nonzero elements.
    // numel_out is set to the number of nonzero elements found.
    virtual CpuStorage nonzero_forward(const Storage& input,
                                       const Shape& input_shape,
                                       Dtype input_dtype,
                                       std::size_t& numel_out) = 0;

    // Optional fused kernels; default implementations call the base backend's
    // linear() followed by relu/gelu and throw if the dtype is unsupported.
    // Backends that can fuse these two ops (e.g. BNNS, ANE) override them.
    virtual Storage fused_linear_relu_forward(
        const Storage& x, const Storage& w, const Storage& b, const Shape& out_shape, Dtype dt) {
        (void)x;
        (void)w;
        (void)b;
        (void)out_shape;
        (void)dt;
        ErrorBuilder("IBackend::fused_linear_relu_forward")
            .not_implemented("fused linear+relu not implemented on this backend");
        return {};
    }

    virtual Storage fused_linear_gelu_forward(
        const Storage& x, const Storage& w, const Storage& b, const Shape& out_shape, Dtype dt) {
        (void)x;
        (void)w;
        (void)b;
        (void)out_shape;
        (void)dt;
        ErrorBuilder("IBackend::fused_linear_gelu_forward")
            .not_implemented("fused linear+gelu not implemented on this backend");
        return {};
    }

    // Compiles and launches a user-provided Metal Shading Language (MSL) kernel.
    // Default implementation throws not_implemented; only GpuBackend overrides this.
    // grid and threads follow Metal's threadgroups / threadsPerThreadgroup convention.
    virtual Storage run_custom_metal_kernel(const std::string& kernel_source,
                                            const std::string& function_name,
                                            const std::vector<Storage>& inputs,
                                            const Shape& output_shape,
                                            Dtype output_dtype,
                                            const std::array<std::size_t, 3>& grid,
                                            const std::array<std::size_t, 3>& threads) {
        (void)kernel_source;
        (void)function_name;
        (void)inputs;
        (void)output_shape;
        (void)output_dtype;
        (void)grid;
        (void)threads;
        ErrorBuilder("IBackend::run_custom_metal_kernel")
            .not_implemented("Metal kernel execution is only supported on the GPU backend");
        return {};
    }

    // Cast elements to a different dtype; shape is unchanged.
    // CPU: element-wise static_cast loop.  GPU: mlx::core::astype.
    virtual Storage astype(const Storage& a, const Shape& shape,
                            Dtype src_dt, Dtype dst_dt) = 0;

    // Moves `src` to MTLResourceStorageModeShared memory so that both CPU and
    // GPU can access it without a copy.  Default is a no-op; GpuBackend overrides.
    virtual Storage to_shared_storage(const Storage& src, const Shape&) { return src; }

    // Parameter bag for LSTM operations.
    //
    // weights is expected in the order [W_ih, W_hh, b_ih, b_hh] per layer;
    // bidirectional layers double the weight count.  When ``proj_size > 0``
    // (the projected-LSTM / LSTMP variant) an additional weight tensor
    // ``W_hr`` of shape ``(proj_size, hidden_size)`` is appended per layer
    // so the order becomes [W_ih, W_hh, b_ih, b_hh, W_hr].  The recurrent
    // weight ``W_hh`` then has its second axis sized ``proj_size`` instead
    // of ``hidden_size`` because the projected hidden state feeds the
    // next time step.  ``c_n`` keeps shape ``hidden_size`` regardless.
    struct LstmOpts {
        int input_size = 0;
        int hidden_size = 0;
        int num_layers = 1;
        int seq_len = 1;
        int batch_size = 1;
        bool batch_first = false;    // if true, input is (batch, seq, input_size)
        bool bidirectional = false;
        bool has_bias = true;
        int proj_size = 0;           // 0 ⇒ standard LSTM; >0 ⇒ projected LSTM
    };

    // LSTM inference forward.  Returns [output, hn, cn].
    // Default implementation throws not_implemented; concrete backends override.
    virtual std::vector<Storage> lstm_forward(const Storage& input,
                                              const Storage& h0,
                                              const Storage& c0,
                                              const std::vector<Storage>& weights,
                                              const LstmOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) {
        (void)input;
        (void)h0;
        (void)c0;
        (void)weights;
        (void)opts;
        (void)out_shape;
        (void)dt;
        ErrorBuilder("IBackend::lstm_forward")
            .not_implemented("LSTM not supported on this backend");
        return {};
    }

    // LSTM training forward; saves gate and cell activations for BPTT.
    // Returns [output, hn, cn, gates_all, cells_all].
    virtual std::vector<Storage> lstm_forward_train(const Storage& input,
                                                    const Storage& h0,
                                                    const Storage& c0,
                                                    const std::vector<Storage>& weights,
                                                    const LstmOpts& opts,
                                                    Dtype dt) {
        (void)input;
        (void)h0;
        (void)c0;
        (void)weights;
        (void)opts;
        (void)dt;
        ErrorBuilder("IBackend::lstm_forward_train")
            .not_implemented("LSTM training not supported on this backend");
        return {};
    }

    // LSTM BPTT backward.  gates_all and cells_all are the saved activations
    // from lstm_forward_train.  Returns [dX, dh0, dc0, dW_ih, dW_hh, db_ih, db_hh].
    virtual std::vector<Storage> lstm_backward(const Storage& grad_output,
                                               const Storage& grad_hn,
                                               const Storage& grad_cn,
                                               const Storage& input,
                                               const Storage& h0,
                                               const std::vector<Storage>& weights,
                                               const Storage& gates_all,
                                               const Storage& cells_all,
                                               const LstmOpts& opts,
                                               Dtype dt) {
        (void)grad_output;
        (void)grad_hn;
        (void)grad_cn;
        (void)input;
        (void)h0;
        (void)weights;
        (void)gates_all;
        (void)cells_all;
        (void)opts;
        (void)dt;
        ErrorBuilder("IBackend::lstm_backward")
            .not_implemented("LSTM backward not supported on this backend");
        return {};
    }
};

}  // namespace backend
}  // namespace lucid
