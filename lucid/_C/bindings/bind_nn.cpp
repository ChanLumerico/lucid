#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Dtype.h"
#include "../core/Generator.h"
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

void register_nn(py::module_& m) {
    m.def("linear", &linear_op, py::arg("x"), py::arg("W"), py::arg("b"),
          "Fused linear: y = x @ W^T + b. Backward returns (dx, dW, db).");

    m.def("layer_norm", &layer_norm_op, py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5,
          "LayerNorm: y = γ(x-μ)/√(σ²+ε) + β. γ/β shape must match trailing dims of x.");

    m.def("rms_norm", &rms_norm_op, py::arg("x"), py::arg("gamma"), py::arg("eps") = 1e-5,
          "RMSNorm: y = γ · x / √(mean(x²)+ε). No mean subtraction, no β.");

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

    m.def("unfold", &unfold_op, py::arg("x"), py::arg("kernel"), py::arg("stride"), py::arg("pad"),
          py::arg("dilation"), "im2col over an N-D input. Returns (B, C·prod(K), prod(O)).");

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

    m.def("lp_normalize", &lp_normalize_op, py::arg("x"), py::arg("ord"), py::arg("axis"),
          py::arg("eps") = 1e-12, "Lp normalize: y = x / max(||x||_p, eps) along `axis`.");

    m.def("global_response_norm", &global_response_norm_op, py::arg("x"), py::arg("gamma"),
          py::arg("beta"), py::arg("eps") = 1e-6,
          "ConvNeXt-v2 GRN: gamma·(x·Nx) + beta·x with Nx = ||x||_2 / mean.");

    // ---------- Loss kernels ----------
    // reduction:  0 = None, 1 = Mean, 2 = Sum
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

    // ---------- Attention ----------
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
            auto r =
                scaled_dot_product_attention_with_weights_op(q, k, v, attn_mask, scale, is_causal);
            return py::make_tuple(r.output, r.weights);
        },
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("attn_mask") = TensorImplPtr{},
        py::arg("scale"), py::arg("is_causal") = false,
        "As scaled_dot_product_attention but also returns the attention "
        "weights (detached — used for visualization/inspection).");

    // ---------- Embedding family ----------
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

    // ---------- Spatial transformer ----------
    m.def("affine_grid", &affine_grid_op, py::arg("theta"), py::arg("N"), py::arg("H"),
          py::arg("W"), py::arg("align_corners") = true,
          "Builds a sampling grid from (N, 2, 3) affine matrices. "
          "Output shape: (N, H, W, 2).");

    m.def("grid_sample", &grid_sample_op, py::arg("input"), py::arg("grid"), py::arg("mode") = 0,
          py::arg("padding_mode") = 0, py::arg("align_corners") = true,
          "2-D grid sampling. mode: 0=bilinear, 1=nearest. "
          "padding_mode: 0=zeros, 1=border.");

    // ---------- Interpolation family (Phase 6-9) ----------
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

    // ---------- Vision (Phase 6-9) ----------
    m.def("one_hot", &one_hot_op, py::arg("input"), py::arg("num_classes"),
          py::arg("dtype") = Dtype::I8,
          "One-hot encode integer indices. Output shape: input.shape + (C,).");
    m.def("rotate", &rotate_op, py::arg("input"), py::arg("angle_deg"), py::arg("cy"),
          py::arg("cx"), "2-D image rotation (nearest-neighbor; no autograd).");
    m.def("bilinear_layer", &bilinear_layer_op, py::arg("x1"), py::arg("x2"), py::arg("weight"),
          py::arg("bias") = TensorImplPtr{},
          "Learned bilinear layer: y = x1 W x2 + b. "
          "x1: (..., D1), x2: (..., D2), W: (Dout, D1, D2), b: (Dout,).");
}

}  // namespace lucid::bindings
