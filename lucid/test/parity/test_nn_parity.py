"""Parity tests for lucid.nn.functional."""

import importlib
import pytest
import numpy as np
import lucid
import lucid.nn.functional as LF
from lucid.test.helpers.parity import check_parity

_REF_BACKEND = "to" "rch"
ref = pytest.importorskip(_REF_BACKEND)
TF = importlib.import_module(_REF_BACKEND + ".nn.functional")


def _pair(shape, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float32)
    return lucid.tensor(data.copy()), ref.tensor(data.copy())


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
        t = ref.tensor(x_np.copy())
        lw = lucid.ones(8)
        tw = ref.ones(8)
        lb = lucid.zeros(8)
        tb = ref.zeros(8)
        check_parity(LF.layer_norm(l, [8], lw, lb), TF.layer_norm(t, [8], tw, tb))

    def test_layer_norm_bias_false(self):
        import lucid.nn as lnn

        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        ref.manual_seed(0)
        t_mod = ref.nn.LayerNorm(8, bias=False)
        weight_np = t_mod.weight.detach().numpy().copy()
        l_mod = lnn.LayerNorm(8, bias=False)
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        new_impl = _ce.TensorImpl(weight_np, _ce.Device.CPU, False)
        l_mod.weight._impl = _iwg(new_impl, l_mod.weight._impl.requires_grad)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )

    def test_layer_norm_elementwise_affine_false(self):
        import lucid.nn as lnn

        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)
        l_mod = lnn.LayerNorm(8, elementwise_affine=False)
        t_mod = ref.nn.LayerNorm(8, elementwise_affine=False)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )

    @pytest.mark.parametrize("affine", [True, False])
    def test_batch_norm_train_running_stats(self, affine):
        import lucid.nn as lnn
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        ref.manual_seed(0)
        t_mod = ref.nn.BatchNorm2d(4, affine=affine)
        l_mod = lnn.BatchNorm2d(4, affine=affine)
        if affine:
            for n, lp in l_mod.named_parameters():
                td_p = dict(t_mod.named_parameters())[n].detach().numpy().copy()
                new_impl = _ce.TensorImpl(td_p, _ce.Device.CPU, False)
                lp._impl = _iwg(new_impl, lp._impl.requires_grad)
        # Run several batches in train mode → running stats accumulate.
        rng = np.random.default_rng(3)
        for _ in range(4):
            x_np = rng.standard_normal((4, 4, 5, 5)).astype(np.float32) + 0.5
            l_mod(lucid.tensor(x_np.copy()))
            t_mod(ref.tensor(x_np.copy()))
        # Compare running stats — must match modulo float rounding.
        check_parity(l_mod.running_mean, t_mod.running_mean)
        check_parity(l_mod.running_var, t_mod.running_var, atol=2e-4)

    def test_batch_norm_eval_uses_running_stats(self):
        import lucid.nn as lnn
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        ref.manual_seed(0)
        t_mod = ref.nn.BatchNorm2d(4)
        l_mod = lnn.BatchNorm2d(4)
        for n, lp in l_mod.named_parameters():
            td_p = dict(t_mod.named_parameters())[n].detach().numpy().copy()
            new_impl = _ce.TensorImpl(td_p, _ce.Device.CPU, False)
            lp._impl = _iwg(new_impl, lp._impl.requires_grad)
        rng = np.random.default_rng(4)
        # Train a few steps to populate running stats.
        for _ in range(3):
            x_np = rng.standard_normal((4, 4, 5, 5)).astype(np.float32) + 0.5
            l_mod(lucid.tensor(x_np.copy()))
            t_mod(ref.tensor(x_np.copy()))
        # Switch to eval and compare outputs on fresh input.
        l_mod.eval()
        t_mod.eval()
        x_np = rng.standard_normal((2, 4, 5, 5)).astype(np.float32)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
            atol=2e-4,
        )

    def test_batch_norm_momentum_none_cumulative(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.BatchNorm2d(4, momentum=None, affine=False)
        l_mod = lnn.BatchNorm2d(4, momentum=None, affine=False)
        rng = np.random.default_rng(5)
        for _ in range(5):
            x_np = rng.standard_normal((4, 4, 5, 5)).astype(np.float32)
            l_mod(lucid.tensor(x_np.copy()))
            t_mod(ref.tensor(x_np.copy()))
        check_parity(l_mod.running_mean, t_mod.running_mean)
        check_parity(l_mod.running_var, t_mod.running_var, atol=2e-4)

    def test_batch_norm_track_running_stats_false(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.BatchNorm2d(4, affine=False, track_running_stats=False)
        l_mod = lnn.BatchNorm2d(4, affine=False, track_running_stats=False)
        rng = np.random.default_rng(6)
        x_np = rng.standard_normal((4, 4, 6, 6)).astype(np.float32)
        # Eval mode without running stats — both should fall back to batch stats.
        l_mod.eval()
        t_mod.eval()
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )

    def test_group_norm(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 8, 4, 4)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = ref.tensor(x_np.copy())
        lw = lucid.ones(8)
        lb = lucid.zeros(8)
        tw = ref.ones(8)
        tb = ref.zeros(8)
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
            ref.tensor(x_np.copy()),
            ref.tensor(w_np.copy()),
            ref.tensor(b_np.copy()),
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
            ref.tensor(x_np.copy()),
            ref.tensor(w_np.copy()),
            ref.tensor(b_np.copy()),
            padding=1,
        )
        check_parity(l_out, t_out)


class TestPadParity:
    """Parity for F.pad across all four modes — backbone of Conv padding_mode."""

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_pad_2d_forward(self, mode):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
        pad = (2, 1, 1, 2)  # (l, r, t, b) on last 2 dims
        l_out = LF.pad(lucid.tensor(x_np.copy()), pad, mode=mode)
        t_out = TF.pad(ref.tensor(x_np.copy()), pad, mode=mode)
        check_parity(l_out, t_out)

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_pad_2d_backward(self, mode):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 3, 6, 6)).astype(np.float32)
        pad = (2, 1, 1, 2)
        xl = lucid.tensor(x_np.copy(), requires_grad=True)
        LF.pad(xl, pad, mode=mode).sum().backward()
        xt = ref.tensor(x_np.copy(), requires_grad=True)
        TF.pad(xt, pad, mode=mode).sum().backward()
        check_parity(xl.grad, xt.grad)

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_pad_1d(self, mode):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 3, 8)).astype(np.float32)
        pad = (2, 3)
        l_out = LF.pad(lucid.tensor(x_np.copy()), pad, mode=mode)
        t_out = TF.pad(ref.tensor(x_np.copy()), pad, mode=mode)
        check_parity(l_out, t_out)


class TestConvPaddingModeParity:
    """Conv*d × every padding_mode forward + backward parity."""

    @staticmethod
    def _make_pair(mod_cls, ref_cls, args, kwargs):
        ref.manual_seed(0)
        t_mod = ref_cls(*args, **kwargs)
        td = {n: p.detach().numpy().copy() for n, p in t_mod.named_parameters()}
        l_mod = mod_cls(*args, **kwargs)
        # Mirror weights into lucid module — using internal _impl swap to dodge
        # the read-only Parameter.data setter.
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        for name, lp in l_mod.named_parameters():
            new_impl = _ce.TensorImpl(td[name], _ce.Device.CPU, False)
            lp._impl = _iwg(new_impl, lp._impl.requires_grad)
        return l_mod, t_mod

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("kernel", [3, 4])
    def test_conv2d_padding_mode_forward(self, mode, kernel):
        import lucid.nn as lnn

        pad = kernel // 2
        l_mod, t_mod = self._make_pair(
            lnn.Conv2d,
            ref.nn.Conv2d,
            (3, 6, kernel),
            {"padding": pad, "padding_mode": mode},
        )
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 3, 7, 7)).astype(np.float32)
        check_parity(l_mod(lucid.tensor(x_np.copy())), t_mod(ref.tensor(x_np.copy())))

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_conv2d_padding_mode_backward(self, mode):
        import lucid.nn as lnn

        l_mod, t_mod = self._make_pair(
            lnn.Conv2d,
            ref.nn.Conv2d,
            (3, 6, 3),
            {"padding": 1, "padding_mode": mode},
        )
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((2, 3, 7, 7)).astype(np.float32)

        xl = lucid.tensor(x_np.copy(), requires_grad=True)
        l_mod(xl).sum().backward()
        xt = ref.tensor(x_np.copy(), requires_grad=True)
        t_mod(xt).sum().backward()

        check_parity(xl.grad, xt.grad)
        check_parity(l_mod.weight.grad, t_mod.weight.grad)
        check_parity(l_mod.bias.grad, t_mod.bias.grad)

    @pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
    def test_conv1d_padding_mode(self, mode):
        import lucid.nn as lnn

        l_mod, t_mod = self._make_pair(
            lnn.Conv1d,
            ref.nn.Conv1d,
            (4, 6, 3),
            {"padding": 1, "padding_mode": mode},
        )
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((2, 4, 9)).astype(np.float32)
        check_parity(l_mod(lucid.tensor(x_np.copy())), t_mod(ref.tensor(x_np.copy())))

    @pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
    def test_conv3d_padding_mode(self, mode):
        import lucid.nn as lnn

        l_mod, t_mod = self._make_pair(
            lnn.Conv3d,
            ref.nn.Conv3d,
            (2, 4, 3),
            {"padding": 1, "padding_mode": mode},
        )
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal((1, 2, 5, 5, 5)).astype(np.float32)
        check_parity(l_mod(lucid.tensor(x_np.copy())), t_mod(ref.tensor(x_np.copy())))


class TestConvSamePaddingParity:
    @staticmethod
    def _make_pair(args, kwargs):
        ref.manual_seed(0)
        import lucid.nn as lnn

        t_mod = ref.nn.Conv2d(*args, **kwargs)
        td = {n: p.detach().numpy().copy() for n, p in t_mod.named_parameters()}
        l_mod = lnn.Conv2d(*args, **kwargs)
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        for name, lp in l_mod.named_parameters():
            new_impl = _ce.TensorImpl(td[name], _ce.Device.CPU, False)
            lp._impl = _iwg(new_impl, lp._impl.requires_grad)
        return l_mod, t_mod

    @pytest.mark.parametrize("kernel", [3, 4, 5])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_same_padding_parity(self, kernel, dilation):
        l_mod, t_mod = self._make_pair(
            (3, 6, kernel),
            {"padding": "same", "dilation": dilation},
        )
        rng = np.random.default_rng(5)
        x_np = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )

    def test_valid_padding_parity(self):
        l_mod, t_mod = self._make_pair((3, 6, 3), {"padding": "valid"})
        rng = np.random.default_rng(6)
        x_np = rng.standard_normal((2, 3, 9, 9)).astype(np.float32)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )


class TestGroupedConvParity:
    @staticmethod
    def _make_pair(args, kwargs):
        ref.manual_seed(0)
        import lucid.nn as lnn

        t_mod = ref.nn.Conv2d(*args, **kwargs)
        td = {n: p.detach().numpy().copy() for n, p in t_mod.named_parameters()}
        l_mod = lnn.Conv2d(*args, **kwargs)
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        for name, lp in l_mod.named_parameters():
            new_impl = _ce.TensorImpl(td[name], _ce.Device.CPU, False)
            lp._impl = _iwg(new_impl, lp._impl.requires_grad)
        return l_mod, t_mod

    @pytest.mark.parametrize("groups", [1, 2, 4, 8])
    def test_grouped_conv2d_forward(self, groups):
        Cin = 8
        Cout = 16
        l_mod, t_mod = self._make_pair(
            (Cin, Cout, 3),
            {"padding": 1, "groups": groups},
        )
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((2, Cin, 7, 7)).astype(np.float32)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )

    @pytest.mark.parametrize("groups", [1, 2, 4, 8])
    def test_grouped_conv2d_backward(self, groups):
        Cin = 8
        Cout = 16
        l_mod, t_mod = self._make_pair(
            (Cin, Cout, 3),
            {"padding": 1, "groups": groups},
        )
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((2, Cin, 7, 7)).astype(np.float32)

        xl = lucid.tensor(x_np.copy(), requires_grad=True)
        l_mod(xl).sum().backward()
        xt = ref.tensor(x_np.copy(), requires_grad=True)
        t_mod(xt).sum().backward()
        check_parity(xl.grad, xt.grad)
        check_parity(l_mod.weight.grad, t_mod.weight.grad)


class TestConvTransposeParity:
    @staticmethod
    def _make_pair(args, kwargs):
        ref.manual_seed(0)
        import lucid.nn as lnn

        t_mod = ref.nn.ConvTranspose2d(*args, **kwargs)
        td = {n: p.detach().numpy().copy() for n, p in t_mod.named_parameters()}
        l_mod = lnn.ConvTranspose2d(*args, **kwargs)
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        for name, lp in l_mod.named_parameters():
            new_impl = _ce.TensorImpl(td[name], _ce.Device.CPU, False)
            lp._impl = _iwg(new_impl, lp._impl.requires_grad)
        return l_mod, t_mod

    @pytest.mark.parametrize(
        "groups",
        [
            1,
            pytest.param(
                2,
                marks=pytest.mark.xfail(
                    reason="conv_transpose2d does not propagate `groups` to the "
                    "engine; tracked separately, out of scope for Conv contract pack",
                    strict=True,
                ),
            ),
        ],
    )
    def test_conv_transpose2d_groups_forward(self, groups):
        l_mod, t_mod = self._make_pair(
            (4, 8, 3),
            {"stride": 2, "padding": 1, "output_padding": 1, "groups": groups},
        )
        rng = np.random.default_rng(9)
        x_np = rng.standard_normal((1, 4, 4, 4)).astype(np.float32)
        check_parity(
            l_mod(lucid.tensor(x_np.copy())),
            t_mod(ref.tensor(x_np.copy())),
        )


class TestPoolParity:
    def test_avg_pool2d(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = ref.tensor(x_np.copy())
        check_parity(LF.avg_pool2d(l, 2, 2), TF.avg_pool2d(t, 2, 2))

    def test_max_pool2d(self):
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 4, 8, 8)).astype(np.float32)
        l = lucid.tensor(x_np.copy())
        t = ref.tensor(x_np.copy())
        check_parity(LF.max_pool2d(l, 2, 2), TF.max_pool2d(t, 2, 2))


class TestLossParity:
    def test_mse_loss(self):
        rng = np.random.default_rng(0)
        p = rng.standard_normal((8,)).astype(np.float32)
        q = rng.standard_normal((8,)).astype(np.float32)
        check_parity(
            LF.mse_loss(lucid.tensor(p.copy()), lucid.tensor(q.copy())),
            TF.mse_loss(ref.tensor(p.copy()), ref.tensor(q.copy())),
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
                ref.tensor(logits.copy()), ref.tensor(targets.astype(np.int64))
            ),
            atol=2e-4,
        )

    @pytest.mark.parametrize("ignore_index", [-1, 2])
    def test_cross_entropy_ignore_index(self, ignore_index):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((8, 5)).astype(np.float32)
        targets = rng.integers(0, 5, size=8).astype(np.int32)
        # Inject ignore_index into a couple positions.
        targets[0] = ignore_index
        targets[3] = ignore_index
        check_parity(
            LF.cross_entropy(
                lucid.tensor(logits.copy()),
                lucid.tensor(targets.copy()),
                ignore_index=ignore_index,
            ),
            TF.cross_entropy(
                ref.tensor(logits.copy()),
                ref.tensor(targets.astype(np.int64)),
                ignore_index=ignore_index,
            ),
            atol=2e-4,
        )

    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.3])
    def test_cross_entropy_label_smoothing(self, smoothing):
        rng = np.random.default_rng(1)
        logits = rng.standard_normal((6, 4)).astype(np.float32)
        targets = rng.integers(0, 4, size=6).astype(np.int32)
        check_parity(
            LF.cross_entropy(
                lucid.tensor(logits.copy()),
                lucid.tensor(targets.copy()),
                label_smoothing=smoothing,
            ),
            TF.cross_entropy(
                ref.tensor(logits.copy()),
                ref.tensor(targets.astype(np.int64)),
                label_smoothing=smoothing,
            ),
            atol=2e-4,
        )

    def test_cross_entropy_weight(self):
        rng = np.random.default_rng(2)
        logits = rng.standard_normal((6, 4)).astype(np.float32)
        targets = rng.integers(0, 4, size=6).astype(np.int32)
        weight = np.array([0.5, 1.0, 2.0, 1.5], dtype=np.float32)
        check_parity(
            LF.cross_entropy(
                lucid.tensor(logits.copy()),
                lucid.tensor(targets.copy()),
                weight=lucid.tensor(weight.copy()),
            ),
            TF.cross_entropy(
                ref.tensor(logits.copy()),
                ref.tensor(targets.astype(np.int64)),
                weight=ref.tensor(weight.copy()),
            ),
            atol=2e-4,
        )

    def test_cross_entropy_combined(self):
        rng = np.random.default_rng(3)
        logits = rng.standard_normal((8, 5)).astype(np.float32)
        targets = rng.integers(0, 5, size=8).astype(np.int32)
        targets[1] = -1
        weight = np.array([0.5, 1.0, 2.0, 1.5, 0.8], dtype=np.float32)
        check_parity(
            LF.cross_entropy(
                lucid.tensor(logits.copy()),
                lucid.tensor(targets.copy()),
                weight=lucid.tensor(weight.copy()),
                ignore_index=-1,
                label_smoothing=0.1,
            ),
            TF.cross_entropy(
                ref.tensor(logits.copy()),
                ref.tensor(targets.astype(np.int64)),
                weight=ref.tensor(weight.copy()),
                ignore_index=-1,
                label_smoothing=0.1,
            ),
            atol=2e-4,
        )

    def test_nll_loss_ignore_index(self):
        rng = np.random.default_rng(4)
        log_q = rng.standard_normal((6, 4)).astype(np.float32)
        log_q = log_q - np.log(np.exp(log_q).sum(axis=1, keepdims=True))
        targets = np.array([0, -1, 2, -1, 1, 3], dtype=np.int32)
        check_parity(
            LF.nll_loss(
                lucid.tensor(log_q.copy()),
                lucid.tensor(targets.copy()),
                ignore_index=-1,
            ),
            TF.nll_loss(
                ref.tensor(log_q.copy()),
                ref.tensor(targets.astype(np.int64)),
                ignore_index=-1,
            ),
            atol=2e-4,
        )

    def test_bce_with_logits_pos_weight(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((8, 4)).astype(np.float32)
        t = rng.integers(0, 2, size=(8, 4)).astype(np.float32)
        pw = np.array([1.5, 2.0, 0.5, 1.0], dtype=np.float32)
        check_parity(
            LF.binary_cross_entropy_with_logits(
                lucid.tensor(x.copy()),
                lucid.tensor(t.copy()),
                pos_weight=lucid.tensor(pw.copy()),
            ),
            TF.binary_cross_entropy_with_logits(
                ref.tensor(x.copy()),
                ref.tensor(t.copy()),
                pos_weight=ref.tensor(pw.copy()),
            ),
            atol=2e-4,
        )

    def test_bce_with_logits_full(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((8, 4)).astype(np.float32)
        t = rng.integers(0, 2, size=(8, 4)).astype(np.float32)
        w = rng.uniform(0.5, 2.0, size=(8, 4)).astype(np.float32)
        pw = np.array([1.5, 2.0, 0.5, 1.0], dtype=np.float32)
        check_parity(
            LF.binary_cross_entropy_with_logits(
                lucid.tensor(x.copy()),
                lucid.tensor(t.copy()),
                weight=lucid.tensor(w.copy()),
                pos_weight=lucid.tensor(pw.copy()),
            ),
            TF.binary_cross_entropy_with_logits(
                ref.tensor(x.copy()),
                ref.tensor(t.copy()),
                weight=ref.tensor(w.copy()),
                pos_weight=ref.tensor(pw.copy()),
            ),
            atol=2e-4,
        )

    def test_bce_with_weight(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(0.05, 0.95, size=(8, 4)).astype(np.float32)
        t = rng.integers(0, 2, size=(8, 4)).astype(np.float32)
        w = rng.uniform(0.5, 2.0, size=(8, 4)).astype(np.float32)
        check_parity(
            LF.binary_cross_entropy(
                lucid.tensor(x.copy()),
                lucid.tensor(t.copy()),
                weight=lucid.tensor(w.copy()),
            ),
            TF.binary_cross_entropy(
                ref.tensor(x.copy()),
                ref.tensor(t.copy()),
                weight=ref.tensor(w.copy()),
            ),
            atol=2e-4,
        )

    def test_kl_div_batchmean(self):
        rng = np.random.default_rng(8)
        log_q = np.log(rng.dirichlet([1.0] * 4, size=4).astype(np.float32))
        p = rng.dirichlet([1.0] * 4, size=4).astype(np.float32)
        check_parity(
            LF.kl_div(
                lucid.tensor(log_q.copy()),
                lucid.tensor(p.copy()),
                reduction="batchmean",
            ),
            TF.kl_div(
                ref.tensor(log_q.copy()), ref.tensor(p.copy()), reduction="batchmean"
            ),
            atol=2e-4,
        )

    def test_kl_div_log_target(self):
        rng = np.random.default_rng(9)
        log_q = np.log(rng.dirichlet([1.0] * 4, size=4).astype(np.float32))
        log_p = np.log(rng.dirichlet([1.0] * 4, size=4).astype(np.float32))
        check_parity(
            LF.kl_div(
                lucid.tensor(log_q.copy()),
                lucid.tensor(log_p.copy()),
                reduction="sum",
                log_target=True,
            ),
            TF.kl_div(
                ref.tensor(log_q.copy()),
                ref.tensor(log_p.copy()),
                reduction="sum",
                log_target=True,
            ),
            atol=2e-4,
        )


class TestMHAParity:
    """Multi-head attention parity vs the reference framework."""

    @staticmethod
    def _mirror(lp, rt):
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        new_impl = _ce.TensorImpl(
            rt.detach().numpy().astype(np.float32), _ce.Device.CPU, False
        )
        lp._impl = _iwg(new_impl, lp._impl.requires_grad)

    def _build_pair(self, **kwargs):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.MultiheadAttention(**kwargs)
        l_mod = lnn.MultiheadAttention(**kwargs)
        ref_sd = t_mod.state_dict()
        if "in_proj_weight" in ref_sd:
            self._mirror(l_mod.in_proj_weight, ref_sd["in_proj_weight"])
            self._mirror(l_mod.in_proj_bias, ref_sd["in_proj_bias"])
        else:
            self._mirror(l_mod.q_proj_weight, ref_sd["q_proj_weight"])
            self._mirror(l_mod.k_proj_weight, ref_sd["k_proj_weight"])
            self._mirror(l_mod.v_proj_weight, ref_sd["v_proj_weight"])
            if "in_proj_bias" in ref_sd:
                self._mirror(l_mod.in_proj_bias, ref_sd["in_proj_bias"])
        self._mirror(l_mod.out_proj_weight, ref_sd["out_proj.weight"])
        if "out_proj.bias" in ref_sd:
            self._mirror(l_mod.out_proj_bias, ref_sd["out_proj.bias"])
        return l_mod, t_mod

    def test_self_attention_forward(self):
        l_mod, t_mod = self._build_pair(embed_dim=16, num_heads=4, batch_first=True)
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((2, 5, 16)).astype(np.float32)
        y_l, _ = l_mod(
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            need_weights=False,
        )
        y_t, _ = t_mod(
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            need_weights=False,
        )
        check_parity(y_l, y_t, atol=1e-5)

    def test_self_attention_with_weights(self):
        l_mod, t_mod = self._build_pair(embed_dim=16, num_heads=4, batch_first=True)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((2, 5, 16)).astype(np.float32)
        y_l, w_l = l_mod(
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            need_weights=True,
            average_attn_weights=False,
        )
        y_t, w_t = t_mod(
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            need_weights=True,
            average_attn_weights=False,
        )
        check_parity(y_l, y_t, atol=1e-5)
        check_parity(w_l, w_t, atol=1e-5)

    def test_cross_attention(self):
        l_mod, t_mod = self._build_pair(embed_dim=16, num_heads=4, batch_first=True)
        rng = np.random.default_rng(2)
        q_np = rng.standard_normal((2, 5, 16)).astype(np.float32)
        kv_np = rng.standard_normal((2, 7, 16)).astype(np.float32)
        y_l, _ = l_mod(
            lucid.tensor(q_np.copy()),
            lucid.tensor(kv_np.copy()),
            lucid.tensor(kv_np.copy()),
            need_weights=False,
        )
        y_t, _ = t_mod(
            ref.tensor(q_np.copy()),
            ref.tensor(kv_np.copy()),
            ref.tensor(kv_np.copy()),
            need_weights=False,
        )
        check_parity(y_l, y_t, atol=1e-5)

    def test_kpm_forward(self):
        l_mod, t_mod = self._build_pair(embed_dim=16, num_heads=4, batch_first=True)
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((2, 5, 16)).astype(np.float32)
        kpm = np.array([[False, False, False, True, True], [False, True, True, True, True]])
        y_l, _ = l_mod(
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            key_padding_mask=lucid.tensor(kpm.copy(), dtype=lucid.bool_),
            need_weights=False,
        )
        y_t, _ = t_mod(
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            key_padding_mask=ref.tensor(kpm.copy()),
            need_weights=False,
        )
        check_parity(y_l, y_t, atol=1e-5)

    def test_attn_mask_2d(self):
        l_mod, t_mod = self._build_pair(embed_dim=16, num_heads=4, batch_first=True)
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal((2, 5, 16)).astype(np.float32)
        am = np.triu(np.ones((5, 5), dtype=bool), k=1)  # causal-style upper triangle
        y_l, _ = l_mod(
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            lucid.tensor(x_np.copy()),
            attn_mask=lucid.tensor(am.copy(), dtype=lucid.bool_),
            need_weights=False,
        )
        y_t, _ = t_mod(
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            ref.tensor(x_np.copy()),
            attn_mask=ref.tensor(am.copy()),
            need_weights=False,
        )
        check_parity(y_l, y_t, atol=1e-5)


class TestLSTMProjSizeParity:
    """LSTMP (projected LSTM) numerical parity with the reference framework."""

    @staticmethod
    def _mirror(lucid_param, ref_t):
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        new_impl = _ce.TensorImpl(
            ref_t.detach().numpy().astype(np.float32), _ce.Device.CPU, False
        )
        lucid_param._impl = _iwg(new_impl, lucid_param._impl.requires_grad)

    def test_proj_size_forward(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        T, B, I, H, P = 5, 2, 4, 8, 3
        t_mod = ref.nn.LSTM(I, H, num_layers=1, bidirectional=False, proj_size=P)
        l_mod = lnn.LSTM(I, H, proj_size=P)
        for pname in (
            "weight_ih_l0",
            "weight_hh_l0",
            "bias_ih_l0",
            "bias_hh_l0",
            "weight_hr_l0",
        ):
            self._mirror(getattr(l_mod, pname), getattr(t_mod, pname))

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((T, B, I)).astype(np.float32)
        y_l, (hn_l, cn_l) = l_mod(lucid.tensor(x_np.copy()))
        y_t, (hn_t, cn_t) = t_mod(ref.tensor(x_np.copy()))
        check_parity(y_l, y_t, atol=1e-5)
        check_parity(hn_l, hn_t, atol=1e-5)
        check_parity(cn_l, cn_t, atol=1e-5)

    def test_proj_size_backward(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        T, B, I, H, P = 4, 2, 3, 6, 2
        t_mod = ref.nn.LSTM(I, H, proj_size=P)
        l_mod = lnn.LSTM(I, H, proj_size=P)
        for pname in (
            "weight_ih_l0",
            "weight_hh_l0",
            "bias_ih_l0",
            "bias_hh_l0",
            "weight_hr_l0",
        ):
            self._mirror(getattr(l_mod, pname), getattr(t_mod, pname))

        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((T, B, I)).astype(np.float32)
        xt = ref.tensor(x_np.copy(), requires_grad=True)
        yt, _ = t_mod(xt)
        yt.sum().backward()
        xl = lucid.tensor(x_np.copy(), requires_grad=True)
        yl, _ = l_mod(xl)
        yl.sum().backward()

        check_parity(xl.grad, xt.grad, atol=1e-5)
        check_parity(l_mod.weight_ih_l0.grad, t_mod.weight_ih_l0.grad, atol=1e-5)
        check_parity(l_mod.weight_hh_l0.grad, t_mod.weight_hh_l0.grad, atol=1e-5)
        check_parity(l_mod.weight_hr_l0.grad, t_mod.weight_hr_l0.grad, atol=1e-5)


class TestLSTMMultiLayerParity:
    """Multi-layer / bidirectional LSTM parity vs reference framework."""

    @staticmethod
    def _mirror_state_dict(l_mod, t_mod):
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        sd = t_mod.state_dict()
        for key, t in sd.items():
            lp = getattr(l_mod, key)
            new_impl = _ce.TensorImpl(
                t.detach().numpy().astype(np.float32), _ce.Device.CPU, False
            )
            lp._impl = _iwg(new_impl, lp._impl.requires_grad)

    def test_multi_layer_forward(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.LSTM(4, 8, num_layers=2)
        l_mod = lnn.LSTM(4, 8, num_layers=2)
        self._mirror_state_dict(l_mod, t_mod)
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((5, 2, 4)).astype(np.float32)
        y_l, (hn_l, cn_l) = l_mod(lucid.tensor(x_np.copy()))
        y_t, (hn_t, cn_t) = t_mod(ref.tensor(x_np.copy()))
        check_parity(y_l, y_t, atol=1e-5)
        check_parity(hn_l, hn_t, atol=1e-5)
        check_parity(cn_l, cn_t, atol=1e-5)

    def test_bidirectional_forward(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.LSTM(4, 8, bidirectional=True)
        l_mod = lnn.LSTM(4, 8, bidirectional=True)
        self._mirror_state_dict(l_mod, t_mod)
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((5, 2, 4)).astype(np.float32)
        y_l, (hn_l, cn_l) = l_mod(lucid.tensor(x_np.copy()))
        y_t, (hn_t, cn_t) = t_mod(ref.tensor(x_np.copy()))
        check_parity(y_l, y_t, atol=1e-5)
        check_parity(hn_l, hn_t, atol=1e-5)
        check_parity(cn_l, cn_t, atol=1e-5)

    def test_multi_layer_bidirectional_forward(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.LSTM(4, 8, num_layers=2, bidirectional=True)
        l_mod = lnn.LSTM(4, 8, num_layers=2, bidirectional=True)
        self._mirror_state_dict(l_mod, t_mod)
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((5, 2, 4)).astype(np.float32)
        y_l, (hn_l, cn_l) = l_mod(lucid.tensor(x_np.copy()))
        y_t, (hn_t, cn_t) = t_mod(ref.tensor(x_np.copy()))
        check_parity(y_l, y_t, atol=1e-5)
        check_parity(hn_l, hn_t, atol=1e-5)
        check_parity(cn_l, cn_t, atol=1e-5)

    def test_multi_layer_proj_size(self):
        import lucid.nn as lnn

        ref.manual_seed(0)
        t_mod = ref.nn.LSTM(4, 8, num_layers=2, proj_size=3)
        l_mod = lnn.LSTM(4, 8, num_layers=2, proj_size=3)
        self._mirror_state_dict(l_mod, t_mod)
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((5, 2, 4)).astype(np.float32)
        y_l, (hn_l, cn_l) = l_mod(lucid.tensor(x_np.copy()))
        y_t, (hn_t, cn_t) = t_mod(ref.tensor(x_np.copy()))
        check_parity(y_l, y_t, atol=1e-5)
        check_parity(hn_l, hn_t, atol=1e-5)
        check_parity(cn_l, cn_t, atol=1e-5)


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
            ref.tensor(q_np.copy()),
            ref.tensor(k_np.copy()),
            ref.tensor(v_np.copy()),
        )
        check_parity(l_out, t_out, atol=2e-4)
