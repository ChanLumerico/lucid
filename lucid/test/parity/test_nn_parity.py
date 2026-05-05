"""
Parity tests: lucid.nn.functional vs torch.nn.functional.
"""

import pytest
import numpy as np
import lucid
import lucid.nn.functional as LF
from lucid.test.helpers.parity import check_parity

torch = pytest.importorskip("torch")
import torch.nn.functional as TF


def _pair(shape, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32)
    return lucid.tensor(data.copy()), torch.tensor(data.copy())


class TestActivationParity:
    @pytest.mark.parametrize(
        "name,kwargs",
        [
            ("relu", {}),
            ("sigmoid", {}),
            ("selu", {}),
            ("softplus", {}),
            ("relu6", {}),
            ("tanh", {}),
            ("mish", {}),
            ("silu", {}),
            ("hardswish", {}),
            ("hardsigmoid", {}),
            ("leaky_relu", {"negative_slope": 0.1}),
            ("elu", {"alpha": 1.0}),
        ],
    )
    def test_activation(self, name, kwargs):
        l, t = _pair((4, 8))
        l_out = getattr(LF, name)(l, **kwargs)
        t_out = getattr(TF, name)(t, **kwargs)
        check_parity(l_out, t_out)

    def test_softmax(self):
        l, t = _pair((4, 8))
        check_parity(LF.softmax(l, dim=-1), TF.softmax(t, dim=-1))

    def test_log_softmax(self):
        l, t = _pair((4, 8))
        check_parity(LF.log_softmax(l, dim=-1), TF.log_softmax(t, dim=-1), atol=2e-4)

    def test_gelu_none(self):
        l, t = _pair((4, 8))
        check_parity(
            LF.gelu(l, approximate="none"), TF.gelu(t, approximate="none"), atol=2e-4
        )

    def test_gelu_tanh(self):
        l, t = _pair((4, 8))
        check_parity(
            LF.gelu(l, approximate="tanh"), TF.gelu(t, approximate="tanh"), atol=2e-4
        )


class TestNormParity:
    def test_layer_norm(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = torch.tensor(x_np.copy())
        lw = lucid.ones(8)
        tw = torch.ones(8)
        lb = lucid.zeros(8)
        tb = torch.zeros(8)
        check_parity(LF.layer_norm(l, [8], lw, lb), TF.layer_norm(t, [8], tw, tb))

    def test_group_norm(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 8, 4, 4)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = torch.tensor(x_np.copy())
        lw = lucid.ones(8)
        lb = lucid.zeros(8)
        tw = torch.ones(8)
        tb = torch.zeros(8)
        check_parity(
            LF.group_norm(l, num_groups=2, weight=lw, bias=lb),
            TF.group_norm(t, 2, tw, tb),
            atol=2e-4,
        )


class TestLinearParity:
    def test_linear_with_bias(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        w_np = rng.standard_normal((6, 8)).astype(np.float32)
        b_np = rng.standard_normal((6,)).astype(np.float32)
        l_out = LF.linear(
            lucid.tensor(x_np.copy()),
            lucid.tensor(w_np.copy()),
            lucid.tensor(b_np.copy()),
        )
        t_out = TF.linear(
            torch.tensor(x_np.copy()),
            torch.tensor(w_np.copy()),
            torch.tensor(b_np.copy()),
        )
        check_parity(l_out, t_out)


class TestConvParity:
    def test_conv2d_basic(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
        w_np = rng.standard_normal((8, 3, 3, 3)).astype(np.float32)
        b_np = rng.standard_normal((8,)).astype(np.float32)
        l_out = LF.conv2d(
            lucid.tensor(x_np.copy()),
            lucid.tensor(w_np.copy()),
            lucid.tensor(b_np.copy()),
            padding=1,
        )
        t_out = TF.conv2d(
            torch.tensor(x_np.copy()),
            torch.tensor(w_np.copy()),
            torch.tensor(b_np.copy()),
            padding=1,
        )
        check_parity(l_out, t_out)


class TestPoolParity:
    def test_avg_pool2d(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = torch.tensor(x_np.copy())
        check_parity(LF.avg_pool2d(l, 2, 2), TF.avg_pool2d(t, 2, 2))

    def test_max_pool2d(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = torch.tensor(x_np.copy())
        check_parity(LF.max_pool2d(l, 2, 2), TF.max_pool2d(t, 2, 2))


class TestLossParity:
    def test_mse_loss(self):
        rng = np.random.default_rng(0)
        p = rng.standard_normal((8,)).astype(np.float32)
        q = rng.standard_normal((8,)).astype(np.float32)
        check_parity(
            LF.mse_loss(lucid.tensor(p.copy()), lucid.tensor(q.copy())),
            TF.mse_loss(torch.tensor(p.copy()), torch.tensor(q.copy())),
        )

    def test_cross_entropy(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((8, 5)).astype(np.float32)
        targets = rng.integers(0, 5, size=8)
        check_parity(
            LF.cross_entropy(
                lucid.tensor(logits.copy()), lucid.tensor(targets.astype(np.int32))
            ),
            TF.cross_entropy(
                torch.tensor(logits.copy()), torch.tensor(targets.astype(np.int64))
            ),
            atol=2e-4,
        )


class TestSDPAParity:
    def test_sdpa_basic(self):
        rng = np.random.default_rng(0)
        q_np = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        k_np = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        v_np = rng.standard_normal((2, 4, 8, 16)).astype(np.float32)
        l_out = LF.scaled_dot_product_attention(
            lucid.tensor(q_np.copy()),
            lucid.tensor(k_np.copy()),
            lucid.tensor(v_np.copy()),
        )
        t_out = TF.scaled_dot_product_attention(
            torch.tensor(q_np.copy()),
            torch.tensor(k_np.copy()),
            torch.tensor(v_np.copy()),
        )
        check_parity(l_out, t_out, atol=2e-4)
