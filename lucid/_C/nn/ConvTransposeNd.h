// lucid/_C/nn/ConvTransposeNd.h
//
// Autograd-aware N-dimensional transposed convolution (deconvolution) for
// $N \in \{1, 2, 3\}$.
//
// A transposed convolution is the *adjoint* (gradient with respect to the
// input) of a standard cross-correlation; it is **not** a true mathematical
// inverse.  Conceptually each input element is multiplied by the kernel and
// the contributions are scattered into the output grid with stride ``s``,
// so values ``> 1`` upsample the spatial axes.
//
// For each spatial axis $i$ the output extent is
//
// $$
// O_i = (S_i - 1)\, s_i - 2 p_i + d_i (K_i - 1) + p^\text{out}_i + 1
// $$
//
// where ``output_padding`` ($p^\text{out}$) disambiguates the otherwise
// many-to-one mapping from input size to output size induced by stride > 1.
//
// Weight layout is ``(C_in, C_out, K_0, ..., K_{N-1})`` — note the **leading
// channel axis is ``C_in``**, the reverse of regular convolution.
//
// CPU backend uses col2im + GEMM; GPU backend uses the MLX transposed-conv
// primitive.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/FuncOp.h"
#include "../core/AmpPolicy.h"
#include "../core/OpSchema.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the N-dimensional transposed convolution.
//
// Backward of this op is itself a regular convolution against the same
// weights, so the implementation reuses the ``ConvNdBackward`` kernels
// for ``dx`` and the matching im2col path for ``dW``.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered op schema (``"conv_transpose1d"`` /
//     ``"conv_transpose2d"`` / ``"conv_transpose3d"``) with
//     ``AmpPolicy::Promote``.
// stride_ : int[N]
//     Per-axis upsampling factor.  Stride ``> 1`` interleaves implicit
//     zeros between input elements.
// pad_ : int[N]
//     Per-axis ``dilation * (K - 1) - padding`` zero-padding removed
//     from the output.
// opad_ : int[N]
//     Per-axis output padding added to one side of the output to
//     disambiguate the inverse-stride shape.  Must be less than
//     ``stride`` along each axis.
template <int N>

class LUCID_API ConvTransposeNdBackward : public FuncOp<ConvTransposeNdBackward<N>, 3> {
public:
    static const OpSchema schema_v1;
    int stride_[N];  // Upsampling stride per axis.
    int pad_[N];     // Input padding removed from output per axis.
    int opad_[N];    // Output padding added to disambiguate output size.

    // Run the forward transposed convolution and attach the backward
    // node.
    //
    // Parameters
    // ----------
    // x : TensorImplPtr
    //     Input batch of shape ``(B, C_in, S_0, ..., S_{N-1})``.
    // W : TensorImplPtr
    //     Filter bank of shape ``(C_in, C_out, K_0, ..., K_{N-1})``.
    //     The leading axis is ``C_in`` — this is the transposed layout
    //     relative to ``ConvNdBackward``.
    // b : TensorImplPtr
    //     Bias of shape ``(C_out,)`` or empty for bias-less variants.
    // stride : array<int, N>
    //     Per-axis upsampling factor.
    // pad : array<int, N>
    //     Per-axis input padding effectively subtracted from each
    //     output extent.
    // opad : array<int, N>
    //     Per-axis output padding added to one side.  Must satisfy
    //     ``opad[i] < stride[i]``.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output of shape ``(B, C_out, O_0, ..., O_{N-1})`` where each
    //     extent is
    //     $O_i = (S_i - 1)\, s_i - 2 p_i + K_i + p^\text{out}_i$
    //     (or with dilation $d_i$ included as $d_i (K_i - 1)$ instead
    //     of $K_i - 1$).
    //
    // Raises
    // ------
    // ShapeMismatch
    //     If the input rank disagrees with ``N + 2``, if ``W``'s
    //     leading axis does not match ``C_in``, if ``b`` is not 1-D of
    //     length ``C_out``, or if a computed output extent is
    //     non-positive.
    // DeviceMismatch
    //     If ``x``, ``W``, ``b`` do not all live on the same device.
    static TensorImplPtr forward(const TensorImplPtr& x,
                                 const TensorImplPtr& W,
                                 const TensorImplPtr& b,
                                 const int (&stride)[N],
                                 const int (&pad)[N],
                                 const int (&opad)[N]);

    // Backward — compute ``[dx, dW, db]`` for the three saved inputs.
    //
    // The gradient w.r.t. ``x`` is a regular ``ConvNd`` against ``W``
    // (the transposed-conv's adjoint is a forward conv).  ``dW`` is
    // obtained via the matching im2col path against the saved input,
    // and ``db`` via a channel-wise reduction.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient with the forward output shape.
    //
    // Returns
    // -------
    // vector<Storage>
    //     Gradients in declaration order: ``dx``, ``dW``, ``db``.
    std::vector<Storage> apply(Storage grad_out) override;
};

using ConvTranspose1dBackward = ConvTransposeNdBackward<1>;
using ConvTranspose2dBackward = ConvTransposeNdBackward<2>;
using ConvTranspose3dBackward = ConvTransposeNdBackward<3>;

// One-dimensional transposed convolution (fractionally-strided conv).
//
// Upsamples a 1-D batch by inserting ``stride - 1`` implicit zeros
// between consecutive input positions, then applies a regular
// cross-correlation against the kernel.  This is the primitive used in
// decoder / generator architectures to lengthen sequences.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C_in, L)``.
// W : TensorImplPtr
//     Filters of shape ``(C_in, C_out, KL)``.
// b : TensorImplPtr
//     Bias of shape ``(C_out,)`` or empty.
// stride_l : int, optional
//     Length-axis upsampling factor.  Default: ``1``.
// pad_l : int, optional
//     Length-axis padding subtracted from the output extent.
//     Default: ``0``.
// opad_l : int, optional
//     Length-axis output padding (one-sided).  Must be ``< stride_l``.
//     Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C_out, L_out)`` where
//     $L_\text{out} = (L - 1)\, s - 2 p + KL + p^\text{out}$.
LUCID_API TensorImplPtr conv_transpose1d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_l = 1,
                                            int pad_l = 0,
                                            int opad_l = 0);

// Two-dimensional transposed convolution (fractionally-strided conv).
//
// The 2-D upsampling primitive used by VAE / GAN decoders, U-Net
// expansion paths, and super-resolution heads.  Equivalent to the
// gradient of a 2-D convolution w.r.t. its input.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C_in, H, W)``.
// W : TensorImplPtr
//     Filters of shape ``(C_in, C_out, KH, KW)``.
// b : TensorImplPtr
//     Bias of shape ``(C_out,)`` or empty.
// stride_h, stride_w : int, optional
//     Per-axis upsampling factor.  Default: ``1``.
// pad_h, pad_w : int, optional
//     Per-axis padding subtracted from the output extent.
//     Default: ``0``.
// opad_h, opad_w : int, optional
//     Per-axis one-sided output padding.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C_out, H_out, W_out)`` with
//     $H_\text{out} = (H - 1) s_h - 2 p_h + KH + p^\text{out}_h$
//     and analogously for $W_\text{out}$.
//
// Notes
// -----
// When ``stride > 1`` and ``kernel_size`` is not divisible by
// ``stride`` the output can exhibit characteristic *checkerboard*
// artefacts.  A common mitigation is to choose
// ``kernel_size = stride * n`` or to substitute bilinear upsampling
// followed by a regular convolution.
LUCID_API TensorImplPtr conv_transpose2d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_h = 1,
                                            int stride_w = 1,
                                            int pad_h = 0,
                                            int pad_w = 0,
                                            int opad_h = 0,
                                            int opad_w = 0);

// Three-dimensional transposed convolution.
//
// The 3-D extension used in volumetric decoders, video generators and
// medical-image synthesis.  All three spatial axes are upsampled
// simultaneously.
//
// Parameters
// ----------
// x : TensorImplPtr
//     Input of shape ``(B, C_in, D, H, W)``.
// W : TensorImplPtr
//     Filters of shape ``(C_in, C_out, KD, KH, KW)``.
// b : TensorImplPtr
//     Bias of shape ``(C_out,)`` or empty.
// stride_d, stride_h, stride_w : int, optional
//     Per-axis upsampling factor.  Default: ``1``.
// pad_d, pad_h, pad_w : int, optional
//     Per-axis padding subtracted from each output extent.
//     Default: ``0``.
// opad_d, opad_h, opad_w : int, optional
//     Per-axis one-sided output padding.  Default: ``0``.
//
// Returns
// -------
// TensorImplPtr
//     Output of shape ``(B, C_out, D_out, H_out, W_out)`` with each
//     output extent following
//     $X_\text{out} = (X - 1) s_x - 2 p_x + K_X + p^\text{out}_x$
//     for $X \in \{D, H, W\}$.
LUCID_API TensorImplPtr conv_transpose3d_op(const TensorImplPtr& x,
                                            const TensorImplPtr& W,
                                            const TensorImplPtr& b,
                                            int stride_d = 1,
                                            int stride_h = 1,
                                            int stride_w = 1,
                                            int pad_d = 0,
                                            int pad_h = 0,
                                            int pad_w = 0,
                                            int opad_d = 0,
                                            int opad_h = 0,
                                            int opad_w = 0);

}  // namespace lucid
