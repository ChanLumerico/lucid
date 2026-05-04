"""
Tests for lucid.nn.functional — covers all functional sub-modules.
"""

import numpy as np
import pytest
import lucid
import lucid.nn.functional as F


# ── Activations ───────────────────────────────────────────────────────────────

class TestActivations:
    def test_relu_positive(self):
        x = lucid.tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(F.relu(x).numpy(), [1.0, 2.0, 3.0], atol=1e-6)

    def test_relu_negative(self):
        x = lucid.tensor([-1.0, -2.0, 0.0])
        np.testing.assert_allclose(F.relu(x).numpy(), [0.0, 0.0, 0.0], atol=1e-6)

    def test_sigmoid_range(self):
        x = lucid.randn(10)
        y = F.sigmoid(x)
        assert (y.numpy() > 0).all() and (y.numpy() < 1).all()

    def test_sigmoid_zero(self):
        x = lucid.tensor([0.0])
        assert abs(float(F.sigmoid(x).item()) - 0.5) < 1e-5

    def test_tanh_range(self):
        x = lucid.randn(10)
        y = F.tanh(x)
        assert (y.numpy() > -1).all() and (y.numpy() < 1).all()

    def test_gelu_shape(self):
        x = lucid.randn(3, 4)
        assert F.gelu(x).shape == (3, 4)

    def test_silu_shape(self):
        x = lucid.randn(2, 5)
        assert F.silu(x).shape == (2, 5)

    def test_softmax_sums_to_one(self):
        x = lucid.randn(4, 6)
        y = F.softmax(x, dim=-1)
        row_sums = y.numpy().sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)

    def test_log_softmax_negative(self):
        x = lucid.randn(3, 5)
        y = F.log_softmax(x, dim=-1)
        assert (y.numpy() <= 0).all()

    def test_leaky_relu_negative_slope(self):
        x = lucid.tensor([-2.0])
        y = F.leaky_relu(x, negative_slope=0.1)
        assert abs(float(y.item()) - (-0.2)) < 1e-5

    def test_elu_negative(self):
        x = lucid.tensor([-1.0])
        y = F.elu(x, alpha=1.0)
        expected = float(np.exp(-1.0) - 1.0)
        assert abs(float(y.item()) - expected) < 1e-5

    def test_mish_shape(self):
        x = lucid.randn(3)
        assert F.mish(x).shape == (3,)

    def test_softplus_shape(self):
        x = lucid.randn(4)
        assert F.softplus(x).shape == (4,)

    def test_hardswish_shape(self):
        x = lucid.randn(3, 3)
        assert F.hardswish(x).shape == (3, 3)

    def test_glu_halves_last_dim(self):
        x = lucid.randn(2, 8)
        y = F.glu(x, dim=-1)
        assert y.shape == (2, 4)

    def test_softmin_sums_to_one(self):
        x = lucid.randn(3, 5)
        y = F.softmin(x, dim=-1)
        row_sums = y.numpy().sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-5)


# ── Linear ────────────────────────────────────────────────────────────────────

class TestLinear:
    def test_linear_no_bias(self):
        x = lucid.randn(4, 3)
        w = lucid.randn(5, 3)
        y = F.linear(x, w)
        assert y.shape == (4, 5)

    def test_linear_with_bias(self):
        x = lucid.randn(2, 4)
        w = lucid.randn(3, 4)
        b = lucid.randn(3)
        y = F.linear(x, w, b)
        assert y.shape == (2, 3)

    def test_linear_correctness(self):
        x = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
        w = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = lucid.tensor([0.1, 0.2])
        y = F.linear(x, w, b)
        expected = np.array([[1.1, 3.2], [2.1, 4.2]])
        np.testing.assert_allclose(y.numpy(), expected, atol=1e-5)


# ── Conv ─────────────────────────────────────────────────────────────────────

class TestConv:
    def test_conv2d_shape(self):
        x = lucid.randn(2, 3, 8, 8)
        w = lucid.randn(16, 3, 3, 3)
        b = lucid.zeros(16)
        y = F.conv2d(x, w, b, padding=1)
        assert y.shape == (2, 16, 8, 8)

    def test_conv2d_no_padding_shape(self):
        x = lucid.randn(1, 1, 5, 5)
        w = lucid.randn(1, 1, 3, 3)
        b = lucid.zeros(1)
        y = F.conv2d(x, w, b)
        assert y.shape == (1, 1, 3, 3)

    def test_conv2d_with_bias(self):
        x = lucid.randn(2, 3, 4, 4)
        w = lucid.randn(8, 3, 1, 1)
        b = lucid.randn(8)
        y = F.conv2d(x, w, b)
        assert y.shape == (2, 8, 4, 4)

    def test_conv1d_shape(self):
        x = lucid.randn(2, 4, 16)
        w = lucid.randn(8, 4, 3)
        b = lucid.zeros(8)
        y = F.conv1d(x, w, b, padding=1)
        assert y.shape == (2, 8, 16)

    def test_conv2d_stride(self):
        x = lucid.randn(1, 1, 8, 8)
        w = lucid.randn(1, 1, 3, 3)
        b = lucid.zeros(1)
        y = F.conv2d(x, w, b, stride=2)
        assert y.shape == (1, 1, 3, 3)

    def test_conv_transpose2d_shape(self):
        x = lucid.randn(1, 8, 4, 4)
        w = lucid.randn(8, 4, 3, 3)
        b = lucid.zeros(4)
        y = F.conv_transpose2d(x, w, b, padding=1)
        assert y.shape == (1, 4, 4, 4)


# ── Pooling ───────────────────────────────────────────────────────────────────

class TestPooling:
    def test_max_pool2d_shape(self):
        x = lucid.randn(2, 4, 8, 8)
        y = F.max_pool2d(x, kernel_size=2, stride=2)
        assert y.shape == (2, 4, 4, 4)

    def test_avg_pool2d_shape(self):
        x = lucid.randn(2, 4, 8, 8)
        y = F.avg_pool2d(x, kernel_size=2, stride=2)
        assert y.shape == (2, 4, 4, 4)

    def test_avg_pool2d_correctness(self):
        x = lucid.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1,1,1,4)
        y = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        assert y.shape == (1, 1, 1, 2)
        np.testing.assert_allclose(y.numpy(), [[[[1.5, 3.5]]]], atol=1e-5)

    def test_adaptive_avg_pool2d(self):
        x = lucid.randn(2, 4, 8, 8)
        y = F.adaptive_avg_pool2d(x, (1, 1))
        assert y.shape == (2, 4, 1, 1)

    def test_adaptive_avg_pool2d_global(self):
        x = lucid.ones(3, 8, 4, 4)
        y = F.adaptive_avg_pool2d(x, (1, 1))
        np.testing.assert_allclose(y.numpy(), np.ones((3, 8, 1, 1)), atol=1e-5)

    def test_max_pool1d_shape(self):
        x = lucid.randn(2, 4, 16)
        y = F.max_pool1d(x, kernel_size=2, stride=2)
        assert y.shape == (2, 4, 8)


# ── Normalization ─────────────────────────────────────────────────────────────

class TestNormalization:
    def test_layer_norm_shape(self):
        x = lucid.randn(2, 4, 8)
        w, b = lucid.ones(8), lucid.zeros(8)
        y = F.layer_norm(x, [8], w, b)
        assert y.shape == (2, 4, 8)

    def test_layer_norm_zero_mean(self):
        x = lucid.randn(4, 16)
        y = F.layer_norm(x, [16], lucid.ones(16), lucid.zeros(16))
        means = y.numpy().mean(axis=-1)
        np.testing.assert_allclose(means, np.zeros(4), atol=1e-4)

    def test_layer_norm_unit_std(self):
        x = lucid.randn(4, 16)
        y = F.layer_norm(x, [16], lucid.ones(16), lucid.zeros(16))
        stds = y.numpy().std(axis=-1)
        np.testing.assert_allclose(stds, np.ones(4), atol=1e-3)

    def test_rms_norm_shape(self):
        x = lucid.randn(2, 8)
        y = F.rms_norm(x, [8], lucid.ones(8))
        assert y.shape == (2, 8)

    def test_group_norm_shape(self):
        x = lucid.randn(2, 8, 4, 4)
        y = F.group_norm(x, num_groups=4)
        assert y.shape == (2, 8, 4, 4)

    def test_batch_norm_inference(self):
        x = lucid.randn(4, 8, 4, 4)
        running_mean = lucid.zeros(8)
        running_var = lucid.ones(8)
        y = F.batch_norm(x, running_mean, running_var, training=False)
        assert y.shape == (4, 8, 4, 4)

    def test_batch_norm_training(self):
        x = lucid.randn(4, 8, 4, 4)
        running_mean = lucid.zeros(8)
        running_var = lucid.ones(8)
        y = F.batch_norm(x, running_mean, running_var, training=True)
        assert y.shape == (4, 8, 4, 4)


# ── Attention ─────────────────────────────────────────────────────────────────

class TestAttention:
    def test_sdpa_shape(self):
        B, H, T, D = 2, 4, 8, 16
        q = lucid.randn(B, H, T, D)
        k = lucid.randn(B, H, T, D)
        v = lucid.randn(B, H, T, D)
        out = F.scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, H, T, D)

    def test_sdpa_with_mask(self):
        B, H, T, D = 1, 2, 4, 8
        q = lucid.randn(B, H, T, D)
        k = lucid.randn(B, H, T, D)
        v = lucid.randn(B, H, T, D)
        # Causal mask: upper triangle = -inf
        mask = lucid.tensor(np.triu(np.full((T, T), float("-inf")), k=1).astype(np.float32))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.shape == (B, H, T, D)

    def test_sdpa_dropout_zero(self):
        B, H, T, D = 1, 1, 4, 8
        q = lucid.randn(B, H, T, D)
        k = lucid.randn(B, H, T, D)
        v = lucid.randn(B, H, T, D)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        assert out.shape == (B, H, T, D)

    def test_sdpa_causal(self):
        B, H, T, D = 1, 1, 4, 8
        q = lucid.randn(B, H, T, D)
        k = lucid.randn(B, H, T, D)
        v = lucid.randn(B, H, T, D)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == (B, H, T, D)


# ── Loss functions ────────────────────────────────────────────────────────────

class TestLossFunctions:
    def test_mse_loss_zero(self):
        x = lucid.randn(4, 3)
        loss = F.mse_loss(x, x)
        assert abs(float(loss.item())) < 1e-6

    def test_mse_loss_known(self):
        x = lucid.tensor([0.0, 0.0])
        t = lucid.tensor([1.0, 1.0])
        loss = F.mse_loss(x, t)
        assert abs(float(loss.item()) - 1.0) < 1e-5

    def test_l1_loss_zero(self):
        x = lucid.randn(3)
        assert abs(float(F.l1_loss(x, x).item())) < 1e-6

    def test_cross_entropy_shape(self):
        logits = lucid.randn(4, 10)
        targets = lucid.tensor([0, 3, 5, 9], dtype=lucid.int32)
        loss = F.cross_entropy(logits, targets)
        assert loss.shape == ()

    def test_cross_entropy_positive(self):
        logits = lucid.randn(8, 5)
        targets = lucid.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=lucid.int32)
        loss = F.cross_entropy(logits, targets)
        assert float(loss.item()) > 0

    def test_binary_cross_entropy_range(self):
        pred = lucid.tensor([0.7, 0.3, 0.9])
        target = lucid.tensor([1.0, 0.0, 1.0])
        loss = F.binary_cross_entropy(pred, target)
        assert float(loss.item()) > 0

    def test_bce_with_logits(self):
        logits = lucid.randn(4)
        target = lucid.tensor([1.0, 0.0, 1.0, 0.0])
        loss = F.binary_cross_entropy_with_logits(logits, target)
        assert float(loss.item()) > 0

    def test_nll_loss_shape(self):
        log_probs = lucid.log(lucid.tensor([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]]))
        targets = lucid.tensor([1, 0], dtype=lucid.int32)
        loss = F.nll_loss(log_probs, targets)
        assert loss.shape == ()

    def test_smooth_l1_zero(self):
        x = lucid.randn(3)
        assert abs(float(F.smooth_l1_loss(x, x).item())) < 1e-6

    def test_huber_loss_shape(self):
        x = lucid.randn(4, 3)
        t = lucid.randn(4, 3)
        loss = F.huber_loss(x, t)
        assert loss.shape == ()

    def test_kl_div_shape(self):
        log_input = F.log_softmax(lucid.randn(4, 5), dim=-1)
        target = F.softmax(lucid.randn(4, 5), dim=-1)
        loss = F.kl_div(log_input, target, reduction="sum")
        assert loss.shape == ()

    def test_loss_reductions(self):
        x = lucid.randn(4)
        t = lucid.randn(4)
        mean = F.mse_loss(x, t, reduction="mean")
        summed = F.mse_loss(x, t, reduction="sum")
        none = F.mse_loss(x, t, reduction="none")
        assert mean.shape == ()
        assert summed.shape == ()
        assert none.shape == (4,)


# ── Dropout ───────────────────────────────────────────────────────────────────

class TestDropout:
    def test_dropout_inference_passthrough(self):
        x = lucid.randn(10)
        y = F.dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6)

    def test_dropout_training_shape(self):
        x = lucid.randn(100)
        y = F.dropout(x, p=0.5, training=True)
        assert y.shape == (100,)

    def test_dropout2d_inference(self):
        x = lucid.randn(2, 4, 4, 4)
        y = F.dropout2d(x, p=0.5, training=False)
        np.testing.assert_allclose(y.numpy(), x.numpy(), atol=1e-6)


# ── Sparse / Embedding ────────────────────────────────────────────────────────

class TestSparse:
    def test_embedding_shape(self):
        weight = lucid.randn(10, 16)
        indices = lucid.tensor([0, 3, 5, 7], dtype=lucid.int64)
        y = F.embedding(indices, weight)
        assert y.shape == (4, 16)

    def test_embedding_lookup_correctness(self):
        weight = lucid.eye(5)
        indices = lucid.tensor([2], dtype=lucid.int64)
        y = F.embedding(indices, weight)
        expected = np.eye(5)[2]
        np.testing.assert_allclose(y.numpy().squeeze(), expected, atol=1e-5)

    def test_one_hot(self):
        indices = lucid.tensor([0, 2, 4], dtype=lucid.int64)
        y = F.one_hot(indices, num_classes=5)
        assert y.shape == (3, 5)
        expected = np.eye(5, dtype=np.int64)[[0, 2, 4]]
        np.testing.assert_allclose(y.numpy(), expected)
