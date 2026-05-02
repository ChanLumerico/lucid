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

struct MatmulOpts {
    bool transA = false;
    bool transB = false;
    int M = 0, K = 0, N = 0;
    std::size_t batch = 1;
};

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

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual Device device() const noexcept = 0;

    virtual Storage from_cpu(CpuStorage cpu, const Shape& shape) = 0;

    virtual Storage zeros(const Shape& shape, Dtype dt) = 0;

    virtual Storage ones(const Shape& shape, Dtype dt) = 0;

    virtual Storage clone(const Storage& src, const Shape& shape, Dtype dt) = 0;

    virtual Storage contiguous(const Storage& src,
                               const Shape& shape,
                               const Stride& stride,
                               std::size_t storage_offset,
                               bool already_contiguous,
                               Dtype dt) = 0;

    virtual Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage
    bitwise_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;
    virtual Storage
    compare_binary(const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) = 0;
    virtual Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;
    virtual Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

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
    virtual Storage relu6(const Storage& a, const Shape& shape, Dtype dt) = 0;

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

    virtual Storage cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage softmax_backward(
        const Storage& z, const Storage& grad_out, const Shape& shape, int axis, Dtype dt) = 0;
    virtual Storage
    reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) = 0;
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
    virtual Storage
    reshape(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) = 0;
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
    virtual Storage
    stack(const std::vector<Storage>& xs, const Shape& input_shape, int axis, Dtype dt) = 0;
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

    virtual Storage
    matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) = 0;
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
    virtual Storage
    linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) = 0;
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

    virtual StoragePair linalg_eigh(const Storage& a,
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

    virtual Storage
    broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) = 0;

    virtual Storage
    repeat(const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) = 0;

    virtual Storage
    tile(const Storage& a, const Shape& shape, Dtype dt, const std::vector<std::int64_t>& reps) = 0;

    virtual Storage
    permute(const Storage& a, const Shape& shape, const std::vector<int>& perm, Dtype dt) = 0;

    virtual Storage pad(const Storage& a,
                        const Shape& shape,
                        Dtype dt,
                        const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                        double constant) = 0;

    virtual Storage pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) = 0;
    virtual Storage rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) = 0;
    virtual Storage
    clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) = 0;

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

    struct ConvNdOpts {
        int N;
        int groups;
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

    virtual Storage embedding_forward(const Storage& weight,
                                      const Storage& indices,
                                      const Shape& weight_shape,
                                      const Shape& indices_shape,
                                      const Shape& out_shape,
                                      int padding_idx,
                                      Dtype dt) = 0;

    virtual Storage embedding_backward(const Storage& grad_out,
                                       const Storage& indices,
                                       const Shape& weight_shape,
                                       const Shape& indices_shape,
                                       int padding_idx,
                                       Dtype dt) = 0;

    virtual Storage
    sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) = 0;

    virtual std::vector<Storage> rope_forward(const Storage& x,
                                              const Storage* position_ids,
                                              const Shape& x_shape,
                                              bool interleaved,
                                              Dtype pos_dtype,
                                              Dtype dt) = 0;

    virtual Storage rope_backward(const Storage& grad_out,
                                  const Storage& saved_cos,
                                  const Storage& saved_sin,
                                  const Shape& x_shape,
                                  bool interleaved,
                                  Dtype dt) = 0;

    struct PoolOpts {
        int K[3];
        int stride[3];
        int pad[3];
        int N;
    };

    virtual std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                                     const Shape& x_shape,
                                                     const Shape& out_shape,
                                                     const PoolOpts& opts,
                                                     Dtype dt) = 0;

    virtual Storage max_pool_nd_backward(const Storage& grad_out,
                                         const Storage& saved_argmax,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

    virtual Storage avg_pool_nd_forward(const Storage& x,
                                        const Shape& x_shape,
                                        const Shape& out_shape,
                                        const PoolOpts& opts,
                                        Dtype dt) = 0;

    virtual Storage avg_pool_nd_backward(const Storage& grad_out,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) = 0;

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

    virtual std::vector<Storage> batch_norm_eval_backward(const Storage& x,
                                                          const Storage& mean,
                                                          const Storage& gamma,
                                                          const Storage& rstd,
                                                          const Storage& grad_out,
                                                          const Shape& x_shape,
                                                          int C,
                                                          int spatial,
                                                          Dtype dt) = 0;

    virtual std::vector<Storage> lp_normalize_forward(
        const Storage& x, const Shape& x_shape, double ord, int axis, double eps, Dtype dt) = 0;

    virtual Storage lp_normalize_backward(const Storage& x,
                                          const Storage& saved_norm,
                                          const Storage& grad_out,
                                          const Shape& x_shape,
                                          double ord,
                                          int axis,
                                          Dtype dt) = 0;

    virtual std::vector<Storage> global_response_norm_forward(const Storage& x,
                                                              const Storage& gamma,
                                                              const Storage& beta,
                                                              const Shape& x_shape,
                                                              double eps,
                                                              Dtype dt) = 0;

    virtual std::vector<Storage> global_response_norm_backward(const Storage& x,
                                                               const Storage& gamma,
                                                               const Storage& beta,
                                                               const Storage& saved_Nx,
                                                               const Storage& grad_out,
                                                               const Shape& x_shape,
                                                               double eps,
                                                               Dtype dt) = 0;

    virtual Storage interpolate_nearest_2d_forward(
        const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) = 0;

    virtual Storage interpolate_nearest_3d_forward(
        const Storage& input, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) = 0;

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

    virtual Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) = 0;

    virtual Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) = 0;

    virtual Storage grid_sample_forward(const Storage& input,
                                        const Storage& grid,
                                        const Shape& in_shape,
                                        const Shape& grid_shape,
                                        int mode,
                                        int padding_mode,
                                        bool align_corners,
                                        Dtype dt) = 0;

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

    virtual Storage one_hot_forward(const Storage& indices,
                                    const Shape& indices_shape,
                                    int num_classes,
                                    Dtype out_dtype) = 0;

    virtual Storage rotate_forward(const Storage& input,
                                   const Shape& shape,
                                   double angle_rad_neg,
                                   double cx,
                                   double cy,
                                   Dtype dt) = 0;

    virtual Storage ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    virtual Storage lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    virtual Storage add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) = 0;

    virtual Storage mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) = 0;

    virtual Storage
    in_range_mask(const Storage& a, const Shape& shape, Dtype dt, double lo, double hi) = 0;

    virtual Storage leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) = 0;

    virtual Storage positive_mask(const Storage& a, const Shape& shape, Dtype dt) = 0;

    virtual Storage reduce_grad_to_shape(const Storage& grad,
                                         const Shape& grad_shape,
                                         const Shape& target_shape,
                                         Dtype dt) = 0;

    virtual Storage broadcast_back_for_reduce(const Storage& grad,
                                              const Shape& grad_shape,
                                              const Shape& input_shape,
                                              const std::vector<int>& axes,
                                              bool keepdims,
                                              Dtype dt) = 0;

    virtual Storage full(const Shape& shape, Dtype dt, double fill_value) = 0;

    virtual Storage eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) = 0;

    virtual Storage
    diag(const Storage& v, const Shape& v_shape, std::int64_t k, Dtype dt, Shape& out_shape) = 0;

    virtual Storage tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) = 0;

    virtual Storage floordiv(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) = 0;

    virtual Storage inner(const Storage& a,
                          const Storage& b,
                          const Shape& a_shape,
                          const Shape& b_shape,
                          const Shape& out_shape,
                          Dtype dt) = 0;

    virtual CpuStorage permute_cpu(const CpuStorage& src,
                                   const Shape& src_shape,
                                   const std::vector<int>& perm,
                                   Dtype dt) = 0;

    virtual Storage tensordot(const Storage& a,
                              const Storage& b,
                              const Shape& a_shape,
                              const Shape& b_shape,
                              const Shape& out_shape,
                              const std::vector<int>& axes_a,
                              const std::vector<int>& axes_b,
                              Dtype dt) = 0;

    virtual Storage where_op(
        const Storage& cond, const Storage& x, const Storage& y, const Shape& shape, Dtype dt) = 0;

    virtual Storage reduce_broadcast(const Storage& grad,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     Dtype dt) = 0;

    virtual Storage histogram_forward(const Storage& input,
                                      const Shape& input_shape,
                                      Dtype input_dtype,
                                      double lo,
                                      double hi,
                                      std::int64_t bins,
                                      bool density) = 0;

    virtual CpuStorage nonzero_forward(const Storage& input,
                                       const Shape& input_shape,
                                       Dtype input_dtype,
                                       std::size_t& numel_out) = 0;

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

    virtual Storage to_shared_storage(const Storage& src, const Shape&) { return src; }

    struct LstmOpts {
        int input_size = 0;
        int hidden_size = 0;
        int num_layers = 1;
        int seq_len = 1;
        int batch_size = 1;
        bool batch_first = false;
        bool bidirectional = false;
        bool has_bias = true;
    };

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
