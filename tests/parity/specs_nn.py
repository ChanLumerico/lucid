"""Specs for nn ops. One representative per kernel family — the harness is
about cross-device parity, not exhaustive shape coverage."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from lucid._C import engine as E

from ._specs import OpSpec


def _conv2d_inputs(rng):
    x = rng.standard_normal(size=(2, 3, 8, 8)).astype("float32")
    w = rng.standard_normal(size=(4, 3, 3, 3)).astype("float32") * 0.1
    b = rng.standard_normal(size=(4,)).astype("float32") * 0.1
    return [x, w, b]


def _linear_inputs(rng):
    x = rng.standard_normal(size=(4, 6)).astype("float32")
    w = rng.standard_normal(size=(8, 6)).astype("float32") * 0.1
    b = rng.standard_normal(size=(8,)).astype("float32") * 0.1
    return [x, w, b]


def _layernorm_inputs(rng):
    x = rng.standard_normal(size=(2, 4, 6)).astype("float32")
    g = rng.standard_normal(size=(6,)).astype("float32") * 0.1 + 1.0
    b = rng.standard_normal(size=(6,)).astype("float32") * 0.1
    return [x, g, b]


SPECS: list[OpSpec] = [
    OpSpec(
        name="nn_linear",
        engine_fn=lambda ts: E.nn.linear(ts[0], ts[1], ts[2]),
        torch_fn=lambda ts: F.linear(ts[0], ts[1], ts[2]),
        input_gen=_linear_inputs,
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_conv2d",
        engine_fn=lambda ts: E.nn.conv2d(ts[0], ts[1], ts[2], 1, 1, 1, 1, 1, 1, 1),
        torch_fn=lambda ts: F.conv2d(ts[0], ts[1], ts[2], stride=1, padding=1, dilation=1, groups=1),
        input_gen=_conv2d_inputs,
        atol=1e-2, rtol=1e-2,
        notes="conv2d: stride=1 pad=1 dil=1 groups=1.",
    ),
    OpSpec(
        name="nn_avg_pool2d",
        engine_fn=lambda ts: E.nn.avg_pool2d(ts[0], 2, 2, 2, 2, 0, 0),
        torch_fn=lambda ts: F.avg_pool2d(ts[0], kernel_size=2, stride=2, padding=0),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_max_pool2d",
        engine_fn=lambda ts: E.nn.max_pool2d(ts[0], 2, 2, 2, 2, 0, 0),
        torch_fn=lambda ts: F.max_pool2d(ts[0], kernel_size=2, stride=2, padding=0),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
        notes="argmax routing inside max_pool — tie-break differences.",
    ),
    OpSpec(
        name="nn_layer_norm",
        engine_fn=lambda ts: E.nn.layer_norm(ts[0], ts[1], ts[2], 1e-5),
        torch_fn=lambda ts: F.layer_norm(ts[0], (6,), ts[1], ts[2], eps=1e-5),
        input_gen=_layernorm_inputs,
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_rms_norm",
        engine_fn=lambda ts: E.nn.rms_norm(ts[0], ts[1], 1e-5),
        torch_fn=lambda ts: F.rms_norm(ts[0], (6,), ts[1], eps=1e-5),
        input_gen=lambda rng: _layernorm_inputs(rng)[:2],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_mse_loss",
        engine_fn=lambda ts: E.nn.mse_loss(ts[0], ts[1], 1),  # 1=mean
        torch_fn=lambda ts: F.mse_loss(ts[0], ts[1], reduction="mean"),
        input_shapes=[(4, 5), (4, 5)],
        atol=1e-4, rtol=1e-4,
    ),

    # Interpolation — both forward and backward exercised on CPU + GPU.
    OpSpec(
        name="nn_interp_bilinear_align_false",
        engine_fn=lambda ts: E.nn.interpolate_bilinear(ts[0], 16, 16, False),
        torch_fn=lambda ts: F.interpolate(ts[0], size=(16, 16),
                                            mode="bilinear", align_corners=False),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_interp_bilinear_align_true",
        engine_fn=lambda ts: E.nn.interpolate_bilinear(ts[0], 16, 16, True),
        torch_fn=lambda ts: F.interpolate(ts[0], size=(16, 16),
                                            mode="bilinear", align_corners=True),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_interp_trilinear",
        engine_fn=lambda ts: E.nn.interpolate_trilinear(ts[0], 8, 16, 16, False),
        torch_fn=lambda ts: F.interpolate(ts[0], size=(8, 16, 16),
                                            mode="trilinear", align_corners=False),
        input_shapes=[(2, 3, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_interp_nearest_2d",
        engine_fn=lambda ts: E.nn.interpolate_nearest_2d(ts[0], 16, 16),
        torch_fn=lambda ts: F.interpolate(ts[0], size=(16, 16), mode="nearest"),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,  # nearest is non-differentiable
    ),
    OpSpec(
        name="nn_interp_nearest_3d",
        engine_fn=lambda ts: E.nn.interpolate_nearest_3d(ts[0], 8, 16, 16),
        torch_fn=lambda ts: F.interpolate(ts[0], size=(8, 16, 16), mode="nearest"),
        input_shapes=[(2, 3, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),

    # Newly migrated to native MLX (no download/upload).
    OpSpec(
        name="nn_unfold_2d",
        engine_fn=lambda ts: E.nn.unfold(ts[0], [3, 3], [1, 1], [1, 1], [1, 1]),
        torch_fn=lambda ts: F.unfold(ts[0], kernel_size=3, padding=1),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_unfold_strided",
        engine_fn=lambda ts: E.nn.unfold(ts[0], [3, 3], [2, 2], [1, 1], [1, 1]),
        torch_fn=lambda ts: F.unfold(ts[0], kernel_size=3, stride=2, padding=1),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_bilinear_layer",
        engine_fn=lambda ts: E.nn.bilinear_layer(ts[0], ts[1], ts[2], ts[3]),
        torch_fn=lambda ts: F.bilinear(ts[0], ts[1], ts[2], ts[3]),
        input_shapes=[(4, 5), (4, 6), (3, 5, 6), (3,)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_drop_block_p_zero",
        engine_fn=lambda ts: E.nn.drop_block(ts[0], 3, 0.0, 1e-6,
                                              E.Generator(0)),
        torch_fn=lambda ts: ts[0],
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
        notes="p=0 collapses to identity — verifies CPU/GPU parity of mask-build path.",
    ),

    # Pool 1D / 3D
    OpSpec(
        name="nn_avg_pool1d",
        engine_fn=lambda ts: E.nn.avg_pool1d(ts[0], 2, 2, 0),
        torch_fn=lambda ts: F.avg_pool1d(ts[0], kernel_size=2, stride=2, padding=0),
        input_shapes=[(2, 3, 16)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_avg_pool3d",
        engine_fn=lambda ts: E.nn.avg_pool3d(ts[0], 2, 2, 2, 2, 2, 2, 0, 0, 0),
        torch_fn=lambda ts: F.avg_pool3d(ts[0], kernel_size=2, stride=2, padding=0),
        input_shapes=[(2, 3, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_max_pool1d",
        engine_fn=lambda ts: E.nn.max_pool1d(ts[0], 2, 2, 0),
        torch_fn=lambda ts: F.max_pool1d(ts[0], kernel_size=2, stride=2, padding=0),
        input_shapes=[(2, 3, 16)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),
    OpSpec(
        name="nn_max_pool3d",
        engine_fn=lambda ts: E.nn.max_pool3d(ts[0], 2, 2, 2, 2, 2, 2, 0, 0, 0),
        torch_fn=lambda ts: F.max_pool3d(ts[0], kernel_size=2, stride=2, padding=0),
        input_shapes=[(2, 3, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),

    # Adaptive pool (uniform divisors)
    OpSpec(
        name="nn_adaptive_avg_pool2d",
        engine_fn=lambda ts: E.nn.adaptive_avg_pool2d(ts[0], 2, 2),
        torch_fn=lambda ts: F.adaptive_avg_pool2d(ts[0], (2, 2)),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_adaptive_max_pool2d",
        engine_fn=lambda ts: E.nn.adaptive_max_pool2d(ts[0], 2, 2),
        torch_fn=lambda ts: F.adaptive_max_pool2d(ts[0], (2, 2)),
        input_shapes=[(2, 3, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),

    # Conv 1D / 3D
    OpSpec(
        name="nn_conv1d",
        engine_fn=lambda ts: E.nn.conv1d(ts[0], ts[1], ts[2], 1, 1, 1, 1),
        torch_fn=lambda ts: F.conv1d(ts[0], ts[1], ts[2], stride=1, padding=1, dilation=1),
        input_gen=lambda rng: [
            rng.standard_normal((2, 3, 16)).astype("float32"),
            rng.standard_normal((4, 3, 3)).astype("float32") * 0.1,
            rng.standard_normal((4,)).astype("float32") * 0.1,
        ],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_conv3d",
        engine_fn=lambda ts: E.nn.conv3d(ts[0], ts[1], ts[2], 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        torch_fn=lambda ts: F.conv3d(ts[0], ts[1], ts[2], stride=1, padding=1, dilation=1),
        input_gen=lambda rng: [
            rng.standard_normal((2, 3, 4, 8, 8)).astype("float32"),
            rng.standard_normal((4, 3, 3, 3, 3)).astype("float32") * 0.1,
            rng.standard_normal((4,)).astype("float32") * 0.1,
        ],
        atol=1e-2, rtol=1e-2,
    ),

    # Conv transpose 1D / 2D / 3D
    OpSpec(
        name="nn_conv_transpose1d",
        engine_fn=lambda ts: E.nn.conv_transpose1d(ts[0], ts[1], ts[2], 1, 0, 0),
        torch_fn=lambda ts: F.conv_transpose1d(ts[0], ts[1], ts[2], stride=1, padding=0),
        input_gen=lambda rng: [
            rng.standard_normal((2, 3, 8)).astype("float32"),
            rng.standard_normal((3, 4, 3)).astype("float32") * 0.1,
            rng.standard_normal((4,)).astype("float32") * 0.1,
        ],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_conv_transpose2d",
        engine_fn=lambda ts: E.nn.conv_transpose2d(ts[0], ts[1], ts[2], 1, 1, 0, 0, 0, 0),
        torch_fn=lambda ts: F.conv_transpose2d(ts[0], ts[1], ts[2], stride=1, padding=0),
        input_gen=lambda rng: [
            rng.standard_normal((2, 3, 8, 8)).astype("float32"),
            rng.standard_normal((3, 4, 3, 3)).astype("float32") * 0.1,
            rng.standard_normal((4,)).astype("float32") * 0.1,
        ],
        atol=1e-2, rtol=1e-2,
    ),
    OpSpec(
        name="nn_conv_transpose3d",
        engine_fn=lambda ts: E.nn.conv_transpose3d(
            ts[0], ts[1], ts[2], 1, 1, 1, 0, 0, 0, 0, 0, 0),
        torch_fn=lambda ts: F.conv_transpose3d(
            ts[0], ts[1], ts[2], stride=1, padding=0),
        input_gen=lambda rng: [
            rng.standard_normal((2, 3, 4, 8, 8)).astype("float32"),
            rng.standard_normal((3, 4, 3, 3, 3)).astype("float32") * 0.1,
            rng.standard_normal((4,)).astype("float32") * 0.1,
        ],
        atol=1e-2, rtol=1e-2,
    ),

    # Norm variants — pure-function batch_norm (train-mode statistics).
    OpSpec(
        name="nn_batch_norm",
        engine_fn=lambda ts: E.nn.batch_norm(ts[0], ts[1], ts[2], 1e-5),
        torch_fn=lambda ts: F.batch_norm(
            ts[0], None, None, ts[1], ts[2], training=True, eps=1e-5),
        input_gen=lambda rng: [
            rng.standard_normal((4, 3, 8, 8)).astype("float32"),
            (rng.standard_normal((3,)) * 0.1 + 1.0).astype("float32"),
            rng.standard_normal((3,)).astype("float32") * 0.1,
        ],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_group_norm",
        engine_fn=lambda ts: E.nn.group_norm(ts[0], ts[1], ts[2], 4, 1e-5),
        torch_fn=lambda ts: F.group_norm(ts[0], 4, ts[1], ts[2], eps=1e-5),
        input_gen=lambda rng: [
            rng.standard_normal((2, 8, 4, 4)).astype("float32"),
            (rng.standard_normal((8,)) * 0.1 + 1.0).astype("float32"),
            rng.standard_normal((8,)).astype("float32") * 0.1,
        ],
        atol=1e-3, rtol=1e-3,
    ),

    # Embedding (lookup)
    OpSpec(
        name="nn_embedding",
        engine_fn=lambda ts: E.nn.embedding(ts[0], ts[1], -1),
        torch_fn=lambda ts: F.embedding(ts[1].long(), ts[0]),
        input_gen=lambda rng: [
            rng.standard_normal((10, 4)).astype("float32"),
            rng.integers(0, 10, size=(2, 3)).astype("int64"),
        ],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
        notes="indices are non-differentiable; test only forward + dE.",
    ),

    # Dropout (p=0 → identity; verifies the path runs)
    OpSpec(
        name="nn_dropout_p_zero",
        engine_fn=lambda ts: E.nn.dropout(ts[0], 0.0, True, E.Generator(0)),
        torch_fn=lambda ts: ts[0],
        input_shapes=[(4, 5)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),

    # Loss functions
    OpSpec(
        name="nn_nll_loss",
        engine_fn=lambda ts: E.nn.nll_loss(ts[0], ts[1], None, 1, -100),
        torch_fn=lambda ts: F.nll_loss(ts[0], ts[1].long(), reduction="mean"),
        input_gen=lambda rng: [
            np.log(np.abs(rng.standard_normal((4, 5))).astype("float32") + 1e-6),
            rng.integers(0, 5, size=(4,)).astype("int64"),
        ],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_cross_entropy",
        engine_fn=lambda ts: E.nn.cross_entropy_loss(ts[0], ts[1], None, 1, 1e-7, -100),
        torch_fn=lambda ts: F.cross_entropy(ts[0], ts[1].long(), reduction="mean"),
        input_gen=lambda rng: [
            rng.standard_normal((4, 5)).astype("float32"),
            rng.integers(0, 5, size=(4,)).astype("int64"),
        ],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_bce_loss",
        engine_fn=lambda ts: E.nn.bce_loss(ts[0], ts[1], ts[2], 1, 1e-7),
        torch_fn=lambda ts: F.binary_cross_entropy(ts[0], ts[1], reduction="mean"),
        input_gen=lambda rng: [
            (rng.uniform(0.05, 0.95, size=(4, 5))).astype("float32"),
            (rng.uniform(0.0, 1.0, size=(4, 5))).astype("float32"),
            np.ones((4, 5), dtype="float32"),  # weight=1 → no-op
        ],
        atol=1e-3, rtol=1e-3,
        skip_grad=True,
        notes="weight is differentiable in engine but ignored by torch — fwd parity only.",
    ),
    OpSpec(
        name="nn_huber_loss",
        engine_fn=lambda ts: E.nn.huber_loss(ts[0], ts[1], 1.0, 1),
        torch_fn=lambda ts: F.huber_loss(ts[0], ts[1], reduction="mean", delta=1.0),
        input_shapes=[(4, 5), (4, 5)],
        atol=1e-4, rtol=1e-4,
    ),

    # Lp normalize
    OpSpec(
        name="nn_lp_normalize",
        engine_fn=lambda ts: E.nn.lp_normalize(ts[0], 2.0, -1, 1e-12),
        torch_fn=lambda ts: F.normalize(ts[0], p=2.0, dim=-1, eps=1e-12),
        input_shapes=[(4, 5)],
        atol=1e-4, rtol=1e-4,
    ),

    # SDPA (basic) — engine signature: q, k, v, mask, scale, is_causal
    OpSpec(
        name="nn_sdpa",
        engine_fn=lambda ts: E.nn.scaled_dot_product_attention(
            ts[0], ts[1], ts[2], None, 1.0 / (16 ** 0.5), False),
        torch_fn=lambda ts: F.scaled_dot_product_attention(ts[0], ts[1], ts[2]),
        input_gen=lambda rng: [
            rng.standard_normal((2, 4, 8, 16)).astype("float32"),
            rng.standard_normal((2, 4, 8, 16)).astype("float32"),
            rng.standard_normal((2, 4, 8, 16)).astype("float32"),
        ],
        atol=1e-3, rtol=1e-3,
    ),

    # Adaptive pool 1D / 3D
    OpSpec(
        name="nn_adaptive_avg_pool1d",
        engine_fn=lambda ts: E.nn.adaptive_avg_pool1d(ts[0], 4),
        torch_fn=lambda ts: F.adaptive_avg_pool1d(ts[0], 4),
        input_shapes=[(2, 3, 16)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_adaptive_avg_pool3d",
        engine_fn=lambda ts: E.nn.adaptive_avg_pool3d(ts[0], 2, 2, 2),
        torch_fn=lambda ts: F.adaptive_avg_pool3d(ts[0], (2, 2, 2)),
        input_shapes=[(2, 3, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
    ),
    OpSpec(
        name="nn_adaptive_max_pool1d",
        engine_fn=lambda ts: E.nn.adaptive_max_pool1d(ts[0], 4),
        torch_fn=lambda ts: F.adaptive_max_pool1d(ts[0], 4),
        input_shapes=[(2, 3, 16)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),
    OpSpec(
        name="nn_adaptive_max_pool3d",
        engine_fn=lambda ts: E.nn.adaptive_max_pool3d(ts[0], 2, 2, 2),
        torch_fn=lambda ts: F.adaptive_max_pool3d(ts[0], (2, 2, 2)),
        input_shapes=[(2, 3, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),

    # Misc loss / norm / dropout / embedding extras
    OpSpec(
        name="nn_bce_with_logits",
        engine_fn=lambda ts: E.nn.bce_with_logits(ts[0], ts[1], ts[2], ts[3], 1),
        torch_fn=lambda ts: F.binary_cross_entropy_with_logits(
            ts[0], ts[1], weight=ts[2], pos_weight=ts[3], reduction="mean"),
        input_gen=lambda rng: [
            rng.standard_normal((4, 5)).astype("float32"),
            rng.uniform(0.0, 1.0, size=(4, 5)).astype("float32"),
            np.ones((4, 5), dtype="float32"),
            np.ones((5,), dtype="float32"),
        ],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
        notes="torch rejects requires_grad on weight/pos_weight; engine grad chain is fine but harness can't drive the comparison.",
    ),
    OpSpec(
        name="nn_one_hot",
        engine_fn=lambda ts: E.nn.one_hot(ts[0], 5, E.Dtype.F32),
        torch_fn=lambda ts: F.one_hot(ts[0].long(), 5).float(),
        input_gen=lambda rng: [
            rng.integers(0, 5, size=(4,)).astype("int64"),
        ],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),
    OpSpec(
        name="nn_alpha_dropout_p_zero",
        engine_fn=lambda ts: E.nn.alpha_dropout(ts[0], 0.0, True, E.Generator(0)),
        torch_fn=lambda ts: ts[0],
        input_shapes=[(4, 5)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),
    OpSpec(
        name="nn_dropoutnd_p_zero",
        engine_fn=lambda ts: E.nn.dropoutnd(ts[0], 0.0, True, E.Generator(0)),
        torch_fn=lambda ts: ts[0],
        input_shapes=[(2, 4, 8, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),
    OpSpec(
        name="nn_drop_path_p_zero",
        engine_fn=lambda ts: E.nn.drop_path(ts[0], 0.0, True, E.Generator(0)),
        torch_fn=lambda ts: ts[0],
        input_shapes=[(4, 8)],
        atol=1e-4, rtol=1e-4,
        skip_grad=True,
    ),

    # batch_norm 1D / 3D pure-function variants
    OpSpec(
        name="nn_batch_norm1d",
        engine_fn=lambda ts: E.nn.batch_norm1d(ts[0], ts[1], ts[2], 1e-5),
        torch_fn=lambda ts: F.batch_norm(
            ts[0], None, None, ts[1], ts[2], training=True, eps=1e-5),
        input_gen=lambda rng: [
            rng.standard_normal((4, 3, 16)).astype("float32"),
            (rng.standard_normal((3,)) * 0.1 + 1.0).astype("float32"),
            rng.standard_normal((3,)).astype("float32") * 0.1,
        ],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_batch_norm3d",
        engine_fn=lambda ts: E.nn.batch_norm3d(ts[0], ts[1], ts[2], 1e-5),
        torch_fn=lambda ts: F.batch_norm(
            ts[0], None, None, ts[1], ts[2], training=True, eps=1e-5),
        input_gen=lambda rng: [
            rng.standard_normal((2, 3, 4, 8, 8)).astype("float32"),
            (rng.standard_normal((3,)) * 0.1 + 1.0).astype("float32"),
            rng.standard_normal((3,)).astype("float32") * 0.1,
        ],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="nn_batch_norm_eval",
        # Engine signature: (x, mean, var, gamma, beta, eps)
        engine_fn=lambda ts: E.nn.batch_norm_eval(
            ts[0], ts[1], ts[2], ts[3], ts[4], 1e-5),
        torch_fn=lambda ts: F.batch_norm(
            ts[0], ts[1], ts[2], ts[3], ts[4], training=False, eps=1e-5),
        input_gen=lambda rng: [
            rng.standard_normal((4, 3, 8, 8)).astype("float32"),  # x
            rng.standard_normal((3,)).astype("float32") * 0.1,    # mean
            (rng.standard_normal((3,)) ** 2 + 0.1).astype("float32"),  # var
            (rng.standard_normal((3,)) * 0.1 + 1.0).astype("float32"),  # gamma
            rng.standard_normal((3,)).astype("float32") * 0.1,    # beta
        ],
        atol=1e-3, rtol=1e-3,
        skip_grad=True,
        notes="running stats are non-trainable; forward parity only.",
    ),

    # Position embeddings (pure functional)
    OpSpec(
        name="nn_sinusoidal_pos_embedding",
        engine_fn=lambda ts: E.nn.sinusoidal_pos_embedding(8, 16, E.Dtype.F32, E.Device.CPU),
        torch_fn=lambda ts: ts[0],  # placeholder — engine builds standalone
        input_gen=lambda rng: [np.zeros((8, 16), dtype="float32")],
        atol=10.0, rtol=10.0,  # shape sanity only
        skip_grad=True,
        skip_gpu=True,
        notes="constructor — checks engine produces output without error.",
    ),
]
