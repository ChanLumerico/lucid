// lucid/_C/bindings/bind_nn.cpp
//
// Registers all neural-network ops on the `lucid._C.engine.nn` sub-module
// (created in bind.cpp).  The sub-module is imported by the Python layer as
// `lucid._C.engine.nn` and re-exported through `lucid.nn.functional`.
//
// Ops covered:
//   - Fully-connected: linear, bilinear_layer
//   - Normalisation: layer_norm, rms_norm, batch_norm{,1d,3d,eval},
//     group_norm, lp_normalize, global_response_norm
//   - Dropout family: dropout, dropoutnd (spatial), alpha_dropout,
//     drop_block, drop_path
//   - Convolution: conv{1,2,3}d, conv_transpose{1,2,3}d, unfold
//   - Pooling: max_pool{1,2,3}d, avg_pool{1,2,3}d,
//     adaptive_{max,avg}_pool{1,2,3}d
//   - Losses: mse_loss, bce_loss, bce_with_logits, cross_entropy_loss,
//     nll_loss, huber_loss
//   - Attention: scaled_dot_product_attention,
//     scaled_dot_product_attention_with_weights
//   - Embeddings: embedding, sinusoidal_pos_embedding, rotary_pos_embedding
//   - Spatial / vision: affine_grid, grid_sample,
//     interpolate_{bilinear,trilinear,nearest_2d,nearest_3d},
//     one_hot, rotate
//   - Recurrent: lstm_forward
//
// Dropout ops accept an optional `generator` keyword (None = default_generator()).
// The generator pointer is resolved at the Python→C++ boundary inside each lambda.
// LSTM's lstm_forward lambda handles optional h0/c0 (None → zeros) and packs
// IBackend::LstmOpts before delegating to lstm_op.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../backend/Dispatcher.h"
#include "../backend/IBackend.h"
#include "../core/Dtype.h"
#include "../core/Generator.h"
#include "../core/Shape.h"
#include "../core/TensorImpl.h"
#include "../nn/AdaptivePool.h"
#include "../nn/Attention.h"
#include "../nn/BatchNorm.h"
#include "../nn/ConvNd.h"
#include "../nn/ConvTransposeNd.h"
#include "../nn/Dropout.h"
#include "../nn/Embedding.h"
#include "../nn/GroupNorm.h"
#include "../nn/Interpolate.h"
#include "../nn/LSTM.h"
#include "../nn/LayerNorm.h"
#include "../nn/Linear.h"
#include "../nn/Loss.h"
#include "../nn/NormExt.h"
#include "../nn/PoolNd.h"
#include "../nn/RMSNorm.h"
#include "../nn/Spatial.h"
#include "../nn/Vision.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers all nn ops on the sub-module `m` (engine.nn).
void register_nn(py::module_& m) {
    // linear delegates directly; its C++ signature already matches pybind11
    // argument conventions (x, W, b → TensorImplPtr).
    m.def("linear", &linear_op, py::arg("x"), py::arg("W"), py::arg("b"),
          "Fused linear: y = x @ W^T + b. Backward returns (dx, dW, db).");

    m.def("layer_norm", &layer_norm_op, py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5,
          "LayerNorm: y = γ(x-μ)/√(σ²+ε) + β. γ/β shape must match trailing dims of x.");

    m.def("rms_norm", &rms_norm_op, py::arg("x"), py::arg("gamma"), py::arg("eps") = 1e-5,
          "RMSNorm: y = γ · x / √(mean(x²)+ε). No mean subtraction, no β.");

    // Dropout ops share a common pattern: py::object gen_obj is accepted as
    // py::none() by default and cast to Generator* only when non-None.  This
    // avoids making Generator a required argument at the Python call site while
    // still allowing reproducible sequences via an explicit Generator instance.
    m.def(
        "dropout",
        [](const TensorImplPtr& a, double p, bool training, py::object gen_obj) {
            Generator* gen = nullptr;
            if (!gen_obj.is_none()) {
                gen = gen_obj.cast<Generator*>();
            }
            return dropout_op(a, p, training, gen);
        },
        py::arg("a"), py::arg("p"), py::arg("training") = true, py::arg("generator") = py::none(),
        "Inverted dropout. Training: y = x · Bernoulli(1-p) / (1-p). "
        "Inference: identity. `generator=None` uses default_generator().");

    // dropoutnd zeroes entire channels rather than individual elements;
    // suitable for convolutional feature maps where spatial correlation would
    // make element-wise dropout ineffective.
    m.def(
        "dropoutnd",
        [](const TensorImplPtr& a, double p, bool training, py::object gen_obj) {
            Generator* gen = nullptr;
            if (!gen_obj.is_none())
                gen = gen_obj.cast<Generator*>();
            return dropoutnd_op(a, p, training, gen);
        },
        py::arg("a"), py::arg("p"), py::arg("training") = true, py::arg("generator") = py::none(),
        "Channel-wise dropout: zeroes whole feature maps. Input is "
        "(B, C, *spatial); mask is (B, C) broadcast over spatial dims.");

    m.def(
        "alpha_dropout",
        [](const TensorImplPtr& a, double p, bool training, py::object gen_obj) {
            Generator* gen = nullptr;
            if (!gen_obj.is_none())
                gen = gen_obj.cast<Generator*>();
            return alpha_dropout_op(a, p, training, gen);
        },
        py::arg("a"), py::arg("p"), py::arg("training") = true, py::arg("generator") = py::none(),
        "SELU-friendly dropout (Klambauer et al., 2017): rescales to "
        "preserve input mean/var.");

    m.def(
        "drop_block",
        [](const TensorImplPtr& a, std::int64_t block_size, double p, double eps,
           py::object gen_obj) {
            Generator* gen = nullptr;
            if (!gen_obj.is_none())
                gen = gen_obj.cast<Generator*>();
            return drop_block_op(a, block_size, p, eps, gen);
        },
        py::arg("a"), py::arg("block_size"), py::arg("p") = 0.1, py::arg("eps") = 1e-7,
        py::arg("generator") = py::none(),
        "Spatial structured dropout (Ghiasi et al., 2018). Drops "
        "block_size×block_size square regions in 4-D feature maps.");

    m.def(
        "drop_path",
        [](const TensorImplPtr& a, double p, bool scale_by_keep, py::object gen_obj) {
            Generator* gen = nullptr;
            if (!gen_obj.is_none())
                gen = gen_obj.cast<Generator*>();
            return drop_path_op(a, p, scale_by_keep, gen);
        },
        py::arg("a"), py::arg("p") = 0.1, py::arg("scale_by_keep") = true,
        py::arg("generator") = py::none(),
        "Stochastic depth — per-sample dropout. Mask shape (B, 1, ..., 1).");

    // Convolution ops: each conv{N}d variant takes separate per-axis stride,
    // padding, and dilation scalars so callers are explicit about anisotropic
    // kernels.  `groups` enables depthwise and grouped convolution.
    // All three share the same per-parameter gradient API (dx, dW, db).
    m.def("conv1d", &conv1d_op, py::arg("x"), py::arg("W"), py::arg("b"), py::arg("stride_l") = 1,
          py::arg("pad_l") = 0, py::arg("dilation_l") = 1, py::arg("groups") = 1,
          "1D convolution. x:(B,C_in,L), W:(C_out,C_in/G,KL), b:(C_out,). "
          "Backward returns (dx, dW, db).");

    m.def("conv2d", &conv2d_op, py::arg("x"), py::arg("W"), py::arg("b"), py::arg("stride_h") = 1,
          py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
          py::arg("dilation_h") = 1, py::arg("dilation_w") = 1, py::arg("groups") = 1,
          "2D convolution. x:(B,C_in,H,W), W:(C_out,C_in/G,KH,KW), b:(C_out,). "
          "Backward returns (dx, dW, db).");

    m.def("conv3d", &conv3d_op, py::arg("x"), py::arg("W"), py::arg("b"), py::arg("stride_d") = 1,
          py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_d") = 0,
          py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("dilation_d") = 1,
          py::arg("dilation_h") = 1, py::arg("dilation_w") = 1, py::arg("groups") = 1,
          "3D convolution. x:(B,C_in,D,H,W), W:(C_out,C_in/G,KD,KH,KW), b:(C_out,). "
          "Backward returns (dx, dW, db).");

    // unfold (im2col) extracts sliding local blocks from the input; the output
    // layout matches reference framework's unfold so it can feed gemm-based conv2d.
    m.def("unfold", &unfold_op, py::arg("x"), py::arg("kernel"), py::arg("stride"), py::arg("pad"),
          py::arg("dilation"), "im2col over an N-D input. Returns (B, C·prod(K), prod(O)).");

    // Transposed convolutions.  opad_{d,h,w} adds extra output padding on the
    // trailing edge to resolve ambiguity when stride > 1 produces multiple
    // valid output sizes.
    m.def("conv_transpose1d", &conv_transpose1d_op, py::arg("x"), py::arg("W"), py::arg("b"),
          py::arg("stride_l") = 1, py::arg("pad_l") = 0, py::arg("opad_l") = 0,
          "1D transposed convolution. x:(B,C_in,L), W:(C_in,C_out,KL), b:(C_out,).");
    m.def("conv_transpose2d", &conv_transpose2d_op, py::arg("x"), py::arg("W"), py::arg("b"),
          py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0,
          py::arg("pad_w") = 0, py::arg("opad_h") = 0, py::arg("opad_w") = 0,
          "2D transposed convolution. x:(B,C_in,H,W), W:(C_in,C_out,KH,KW).");
    m.def("conv_transpose3d", &conv_transpose3d_op, py::arg("x"), py::arg("W"), py::arg("b"),
          py::arg("stride_d") = 1, py::arg("stride_h") = 1, py::arg("stride_w") = 1,
          py::arg("pad_d") = 0, py::arg("pad_h") = 0, py::arg("pad_w") = 0, py::arg("opad_d") = 0,
          py::arg("opad_h") = 0, py::arg("opad_w") = 0,
          "3D transposed convolution. x:(B,C_in,D,H,W), W:(C_in,C_out,KD,KH,KW).");

    // Adaptive pooling: output size is specified directly; the kernel and
    // stride are computed internally assuming uniform spatial partitioning.
    m.def("adaptive_max_pool1d", &adaptive_max_pool1d_op, py::arg("x"), py::arg("output_l"),
          "1D adaptive max pooling (uniform case: L must be divisible by output_l).");
    m.def("adaptive_max_pool2d", &adaptive_max_pool2d_op, py::arg("x"), py::arg("output_h"),
          py::arg("output_w"), "2D adaptive max pooling (uniform case).");
    m.def("adaptive_max_pool3d", &adaptive_max_pool3d_op, py::arg("x"), py::arg("output_d"),
          py::arg("output_h"), py::arg("output_w"), "3D adaptive max pooling (uniform case).");
    m.def("adaptive_avg_pool1d", &adaptive_avg_pool1d_op, py::arg("x"), py::arg("output_l"),
          "1D adaptive average pooling (uniform case).");
    m.def("adaptive_avg_pool2d", &adaptive_avg_pool2d_op, py::arg("x"), py::arg("output_h"),
          py::arg("output_w"), "2D adaptive average pooling (uniform case).");
    m.def("adaptive_avg_pool3d", &adaptive_avg_pool3d_op, py::arg("x"), py::arg("output_d"),
          py::arg("output_h"), py::arg("output_w"), "3D adaptive average pooling (uniform case).");

    // Fixed-kernel pooling: stride=0 is a sentinel that the C++ op interprets
    // as stride == kernel_size (non-overlapping, stride-equals-kernel pooling).
    m.def("max_pool1d", &max_pool1d_op, py::arg("x"), py::arg("kernel_l"), py::arg("stride_l") = 0,
          py::arg("pad_l") = 0, "1D max pooling. stride=0 means stride==kernel.");
    m.def("max_pool2d", &max_pool2d_op, py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h") = 0, py::arg("stride_w") = 0, py::arg("pad_h") = 0,
          py::arg("pad_w") = 0, "2D max pooling. stride=0 means stride==kernel.");
    m.def("max_pool3d", &max_pool3d_op, py::arg("x"), py::arg("kernel_d"), py::arg("kernel_h"),
          py::arg("kernel_w"), py::arg("stride_d") = 0, py::arg("stride_h") = 0,
          py::arg("stride_w") = 0, py::arg("pad_d") = 0, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
          "3D max pooling. stride=0 means stride==kernel.");

    m.def("avg_pool1d", &avg_pool1d_op, py::arg("x"), py::arg("kernel_l"), py::arg("stride_l") = 0,
          py::arg("pad_l") = 0, "1D average pooling. stride=0 means stride==kernel.");
    m.def("avg_pool2d", &avg_pool2d_op, py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h") = 0, py::arg("stride_w") = 0, py::arg("pad_h") = 0,
          py::arg("pad_w") = 0, "2D average pooling. stride=0 means stride==kernel.");
    m.def("avg_pool3d", &avg_pool3d_op, py::arg("x"), py::arg("kernel_d"), py::arg("kernel_h"),
          py::arg("kernel_w"), py::arg("stride_d") = 0, py::arg("stride_h") = 0,
          py::arg("stride_w") = 0, py::arg("pad_d") = 0, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
          "3D average pooling. stride=0 means stride==kernel.");

    // Normalisation ops.  The "pure-function" batch norm variants compute
    // mean/var from the current mini-batch (training statistics only); the
    // Module wrapper in lucid.nn manages running_mean / running_var separately.
    m.def("batch_norm1d", &batch_norm1d_op, py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5, "Pure-function BatchNorm1d. x:(B,C,L). γ,β:(C,).");
    m.def("batch_norm", &batch_norm_op, py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5,
          "Pure-function BatchNorm2d (train-mode statistics, no running stats). "
          "x:(B,C,H,W). γ,β:(C,). Module wrapper handles running stats.");
    m.def("batch_norm3d", &batch_norm3d_op, py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5, "Pure-function BatchNorm3d. x:(B,C,D,H,W). γ,β:(C,).");

    m.def("group_norm", &group_norm_op, py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("num_groups"), py::arg("eps") = 1e-5,
          "GroupNorm. Channels split into num_groups; mean/var across (C/G,H,W). "
          "InstanceNorm == GroupNorm with num_groups=C.");

    m.def("batch_norm_eval", &batch_norm_eval_op, py::arg("x"), py::arg("mean"), py::arg("var"),
          py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5,
          "Inference-mode BatchNorm using precomputed running stats.");

    m.def("lp_normalize", &lp_normalize_op, py::arg("x"), py::arg("ord"), py::arg("dim"),
          py::arg("eps") = 1e-12, "Lp normalize: y = x / max(||x||_p, eps) along `dim`.");

    m.def("global_response_norm", &global_response_norm_op, py::arg("x"), py::arg("gamma"),
          py::arg("beta"), py::arg("eps") = 1e-6,
          "ConvNeXt-v2 GRN: gamma·(x·Nx) + beta·x with Nx = ||x||_2 / mean.");

    // Loss functions.  The `reduction` argument uses integer codes rather than
    // a string to avoid std::string allocation on every forward call:
    //   0 = None (return per-element tensor)
    //   1 = Mean  (scalar)
    //   2 = Sum   (scalar)
    // Optional TensorImplPtr arguments (weight, pos_weight) default to a null
    // shared_ptr which the C++ op treats as "no per-class weighting".
    m.def("mse_loss", &mse_loss_op, py::arg("input"), py::arg("target"), py::arg("reduction") = 1,
          "MSE loss. reduction: 0=None, 1=Mean, 2=Sum.");
    m.def("bce_loss", &bce_loss_op, py::arg("input"), py::arg("target"),
          py::arg("weight") = TensorImplPtr{}, py::arg("reduction") = 1, py::arg("eps") = 1e-7,
          "Binary cross-entropy. Input must be in [0, 1].");
    m.def("bce_with_logits", &bce_with_logits_op, py::arg("input"), py::arg("target"),
          py::arg("weight") = TensorImplPtr{}, py::arg("pos_weight") = TensorImplPtr{},
          py::arg("reduction") = 1,
          "Binary cross-entropy with sigmoid fused into the loss for stability.");
    m.def("cross_entropy_loss", &cross_entropy_op, py::arg("input"), py::arg("target"),
          py::arg("weight") = TensorImplPtr{}, py::arg("reduction") = 1, py::arg("eps") = 1e-7,
          py::arg("ignore_index") = -100, "Cross-entropy = LogSoftmax + NLL fused.");
    m.def("nll_loss", &nll_loss_op, py::arg("input"), py::arg("target"),
          py::arg("weight") = TensorImplPtr{}, py::arg("reduction") = 1,
          py::arg("ignore_index") = -100,
          "Negative log-likelihood loss; input must be log-probabilities.");
    m.def("huber_loss", &huber_loss_op, py::arg("input"), py::arg("target"), py::arg("delta") = 1.0,
          py::arg("reduction") = 1, "Huber loss with parameter `delta`.");

    // Attention ops.  scaled_dot_product_attention returns only the context
    // tensor (grad-capable); the _with_weights variant also returns the softmax
    // attention weight matrix, but that second output is detached (no autograd)
    // and is intended for visualisation/inspection only.  The lambda unpacks
    // the 2-element std::vector returned by the C++ op into a Python tuple.
    m.def("scaled_dot_product_attention", &scaled_dot_product_attention_op, py::arg("query"),
          py::arg("key"), py::arg("value"), py::arg("attn_mask") = TensorImplPtr{},
          py::arg("scale"), py::arg("is_causal") = false,
          "Fused scaled-dot-product attention. Q/K/V: [..., L, d]. "
          "attn_mask is None, [L_q, L_k], or [B*, L_q, L_k] — Bool masks "
          "use -inf, float masks are added.");
    m.def(
        "scaled_dot_product_attention_with_weights",
        [](const TensorImplPtr& q, const TensorImplPtr& k, const TensorImplPtr& v,
           const TensorImplPtr& attn_mask, double scale, bool is_causal) {
            // The C++ op returns {context, weights}; unpack to a 2-tuple.
            auto out =
                scaled_dot_product_attention_with_weights_op(q, k, v, attn_mask, scale, is_causal);
            return py::make_tuple(out.at(0), out.at(1));
        },
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("attn_mask") = TensorImplPtr{},
        py::arg("scale"), py::arg("is_causal") = false,
        "As scaled_dot_product_attention but also returns the attention "
        "weights (detached — used for visualization/inspection).");

    // Embedding lookup and positional encoding utilities.
    m.def("embedding", &embedding_op, py::arg("weight"), py::arg("indices"),
          py::arg("padding_idx") = -1,
          "Index-gather rows from weight ([N, D]) by indices. "
          "padding_idx (-1 = none) zeroes those rows in the output and "
          "blocks gradient flow.");

    m.def("sinusoidal_pos_embedding", &sinusoidal_pos_embedding_op, py::arg("seq_len"),
          py::arg("embed_dim"), py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          "Standard sinusoidal positional table — no autograd.");

    m.def("rotary_pos_embedding", &rotary_pos_embedding_op, py::arg("input"),
          py::arg("position_ids") = TensorImplPtr{}, py::arg("interleaved") = true,
          "RoPE: rotates last 2 dims (L, D) of input by frequencies "
          "derived from positions. interleaved=True uses (even, odd) pairs; "
          "False splits into two halves.");

    // Spatial transformation and interpolation ops.
    m.def("affine_grid", &affine_grid_op, py::arg("theta"), py::arg("N"), py::arg("H"),
          py::arg("W"), py::arg("align_corners") = true,
          "Builds a sampling grid from (N, 2, 3) affine matrices. "
          "Output shape: (N, H, W, 2).");

    m.def("grid_sample", &grid_sample_op, py::arg("input"), py::arg("grid"), py::arg("mode") = 0,
          py::arg("padding_mode") = 0, py::arg("align_corners") = true,
          "2-D grid sampling. mode: 0=bilinear, 1=nearest. "
          "padding_mode: 0=zeros, 1=border.");

    m.def("interpolate_bilinear", &interpolate_bilinear_op, py::arg("input"), py::arg("H_out"),
          py::arg("W_out"), py::arg("align_corners") = false,
          "2-D bilinear interpolation. Input: (N, C, H, W).");
    m.def("interpolate_trilinear", &interpolate_trilinear_op, py::arg("input"), py::arg("D_out"),
          py::arg("H_out"), py::arg("W_out"), py::arg("align_corners") = false,
          "3-D trilinear interpolation. Input: (N, C, D, H, W).");
    m.def("interpolate_nearest_2d", &interpolate_nearest_2d_op, py::arg("input"), py::arg("H_out"),
          py::arg("W_out"), "2-D nearest-neighbor interpolation (no autograd).");
    m.def("interpolate_nearest_3d", &interpolate_nearest_3d_op, py::arg("input"), py::arg("D_out"),
          py::arg("H_out"), py::arg("W_out"), "3-D nearest-neighbor interpolation (no autograd).");

    m.def("one_hot", &one_hot_op, py::arg("input"), py::arg("num_classes"),
          py::arg("dtype") = Dtype::I8,
          "One-hot encode integer indices. Output shape: input.shape + (C,).");
    m.def("rotate", &rotate_op, py::arg("input"), py::arg("angle_deg"), py::arg("cy"),
          py::arg("cx"), "2-D image rotation (nearest-neighbor; no autograd).");
    m.def("bilinear_layer", &bilinear_layer_op, py::arg("x1"), py::arg("x2"), py::arg("weight"),
          py::arg("bias") = TensorImplPtr{},
          "Learned bilinear layer: y = x1 W x2 + b. "
          "x1: (..., D1), x2: (..., D2), W: (Dout, D1, D2), b: (Dout,).");

    // fold (col2im): inverse of unfold/im2col.
    // Input: (N, C*kH*kW, L) → Output: (N, C, outH, outW).
    m.def(
        "fold",
        [](const TensorImplPtr& x, std::vector<int> output_size, std::vector<int> kernel_size,
           std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation) {
            return fold_op(x, output_size, kernel_size, stride, padding, dilation);
        },
        py::arg("x"), py::arg("output_size"), py::arg("kernel_size"),
        py::arg("stride") = std::vector<int>{1, 1},
        py::arg("padding") = std::vector<int>{0, 0},
        py::arg("dilation") = std::vector<int>{1, 1},
        "col2im: inverse of unfold. (N, C*kH*kW, L) → (N, C, outH, outW).\n"
        "CPU: scatter-add loop.  GPU: CPU fallback.");

    // embedding_bag: pooled embedding lookup.
    // mode: 0=sum, 1=mean, 2=max.
    m.def(
        "ctc_loss",
        [](const TensorImplPtr& log_probs, const TensorImplPtr& targets,
           const TensorImplPtr& input_lengths, const TensorImplPtr& target_lengths,
           int blank, bool zero_infinity) {
            return ctc_loss_op(log_probs, targets, input_lengths, target_lengths,
                               blank, zero_infinity);
        },
        py::arg("log_probs"), py::arg("targets"),
        py::arg("input_lengths"), py::arg("target_lengths"),
        py::arg("blank") = 0, py::arg("zero_infinity") = false,
        "CTC loss. log_probs:(T,N,C), targets:(N*S,) int32, lengths:(N,) int32.\n"
        "Returns per-sample losses (N,). CPU: forward DP in log-domain. GPU: CPU fallback.");

    m.def(
        "embedding_bag",
        [](const TensorImplPtr& weight, const TensorImplPtr& indices,
           const TensorImplPtr& offsets, int mode, int padding_idx, bool include_last_offset) {
            return embedding_bag_op(weight, indices, offsets, mode, padding_idx, include_last_offset);
        },
        py::arg("weight"), py::arg("indices"), py::arg("offsets"),
        py::arg("mode") = 0, py::arg("padding_idx") = -1, py::arg("include_last_offset") = false,
        "Pooled embedding lookup with offset-delimited bags.\n"
        "mode: 0=sum, 1=mean, 2=max.\n"
        "CPU: gather+reduce loop.  GPU: MLX gather+scatter_add/scatter_max.");

    // lstm_forward is implemented as a lambda rather than a direct C++ op
    // pointer because it needs to:
    //   1. Validate and reshape the input to extract seq_len / batch / input_size.
    //   2. Allocate zero h0 / c0 tensors on the correct device when the caller
    //      passes None (common for the initial hidden state).
    //   3. Pack IBackend::LstmOpts from the keyword arguments before calling
    //      lstm_op.
    //   4. Unpack the returned 3-element vector into a Python (output, h_n, c_n)
    //      tuple via py::make_tuple.
    m.def(
        "lstm_forward",
        [](const TensorImplPtr& input, py::object h0_obj, py::object c0_obj,
           const std::vector<TensorImplPtr>& weight_tensors, int hidden_size, int num_layers,
           bool batch_first, bool bidirectional, bool has_bias, int proj_size) -> py::tuple {
            if (!input)
                throw std::invalid_argument("lstm_forward: null input");

            const auto& in_shape = input->shape();
            if (in_shape.size() < 2)
                throw std::invalid_argument("lstm_forward: input must be at least 2-D");

            // Derive sequence/batch dimensions based on batch_first layout.
            const int seq_len =
                batch_first ? static_cast<int>(in_shape[1]) : static_cast<int>(in_shape[0]);
            const int batch =
                batch_first ? static_cast<int>(in_shape[0]) : static_cast<int>(in_shape[1]);
            const int input_size = static_cast<int>(in_shape.back());
            const int num_dirs = bidirectional ? 2 : 1;

            const Dtype dt = input->dtype();
            const Device dev = input->device();

            // Allocate zeros for h0 / c0 when the caller passes None.
            // h0 shape: (num_layers * num_dirs, batch, Hrec) where Hrec is
            // the recurrent (possibly projected) hidden dim.  c0 always
            // uses the cell-state dim (hidden_size).
            const int Hrec = (proj_size > 0) ? proj_size : hidden_size;
            auto make_zeros = [&](int rows, int last) -> TensorImplPtr {
                Shape s{static_cast<std::int64_t>(rows), static_cast<std::int64_t>(batch),
                        static_cast<std::int64_t>(last)};
                auto st = backend::Dispatcher::for_device(dev).zeros(s, dt);
                return std::make_shared<TensorImpl>(std::move(st), s, dt, dev, false);
            };
            TensorImplPtr h0 = h0_obj.is_none() ? make_zeros(num_layers * num_dirs, Hrec)
                                                : h0_obj.cast<TensorImplPtr>();
            TensorImplPtr c0 = c0_obj.is_none() ? make_zeros(num_layers * num_dirs, hidden_size)
                                                : c0_obj.cast<TensorImplPtr>();

            backend::IBackend::LstmOpts opts;
            opts.input_size = input_size;
            opts.hidden_size = hidden_size;
            opts.num_layers = num_layers;
            opts.seq_len = seq_len;
            opts.batch_size = batch;
            opts.batch_first = batch_first;
            opts.bidirectional = bidirectional;
            opts.has_bias = has_bias;
            opts.proj_size = proj_size;

            // Multi-layer / bidirectional + proj_size is not yet supported
            // by the hand-rolled CPU kernel (which is single-layer
            // unidirectional only).  Surface this explicitly so callers see
            // a clean error rather than a downstream BLAS shape mismatch.
            if (proj_size > 0 && (num_layers != 1 || bidirectional))
                throw std::invalid_argument(
                    "lstm_forward: proj_size > 0 is currently supported only "
                    "for num_layers=1 and bidirectional=False.");

            auto results = lstm_op(input, h0, c0, weight_tensors, opts);
            return py::make_tuple(results[0], results[1], results[2]);
        },
        py::arg("input"), py::arg("h0") = py::none(), py::arg("c0") = py::none(),
        py::arg("weights"), py::arg("hidden_size"), py::arg("num_layers") = 1,
        py::arg("batch_first") = false, py::arg("bidirectional") = false,
        py::arg("has_bias") = true, py::arg("proj_size") = 0,
        "LSTM forward with autograd support.\n"
        "Inference (no requires_grad): uses BNNS fast path; falls back to BLAS "
        "when proj_size > 0 (BNNS has no LSTMP support).\n"
        "Training  (requires_grad=True): uses BLAS path + saves gates/cells for BPTT.\n"
        "Returns (output, h_n, c_n).  weights = [wih, whh, bih, bhh] for the standard\n"
        "LSTM, with one extra W_hr (proj_size, hidden_size) appended when proj_size > 0.");
}

}  // namespace lucid::bindings
