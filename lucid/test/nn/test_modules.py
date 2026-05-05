"""Tests for nn.Module API: parameters, state_dict, train/eval, hooks, to()."""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
from lucid.test.helpers.numerics import make_tensor


class SimpleMLP(nn.Module):
    def __init__(self, in_f=8, hid=16, out_f=4):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hid)
        self.fc2 = nn.Linear(hid, out_f)

    def forward(self, x):
        import lucid.nn.functional as F

        return self.fc2(F.relu(self.fc1(x)))


class ExtraStateModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = 0

    def forward(self, x):
        return x

    def get_extra_state(self):
        return {"value": self.value}

    def set_extra_state(self, state):
        self.value = state["value"]


class TestParameters:
    def test_parameters_count(self):
        model = SimpleMLP()
        params = list(model.parameters())
        assert len(params) == 4  # fc1.w, fc1.b, fc2.w, fc2.b

    def test_named_parameters(self):
        model = SimpleMLP()
        names = [n for n, _ in model.named_parameters()]
        assert "fc1.weight" in names
        assert "fc1.bias" in names

    def test_parameters_require_grad(self):
        model = SimpleMLP()
        for p in model.parameters():
            assert p.requires_grad

    def test_zero_grad(self):
        model = SimpleMLP()
        x = make_tensor((3, 8))
        out = model(x)
        lucid.sum(out).backward()
        model.zero_grad()
        for p in model.parameters():
            assert p.grad is None


class TestStateDict:
    def test_state_dict_has_keys(self):
        model = SimpleMLP()
        sd = model.state_dict()
        assert "fc1.weight" in sd
        assert "fc2.bias" in sd

    def test_load_state_dict_roundtrip(self):
        model = SimpleMLP()
        sd = model.state_dict()
        model2 = SimpleMLP()
        result = model2.load_state_dict(sd)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        # Weights should match
        for k in sd:
            np.testing.assert_array_almost_equal(
                model2.state_dict()[k].numpy(), sd[k].numpy()
            )

    def test_load_state_dict_size_mismatch_raises(self):
        model = nn.Linear(4, 2)
        sd = model.state_dict()
        sd["weight"] = lucid.ones((3, 4))
        with pytest.raises(RuntimeError, match="size mismatch"):
            model.load_state_dict(sd, strict=False)

    def test_load_state_dict_reports_missing_unexpected(self):
        model = nn.Linear(4, 2)
        sd = model.state_dict()
        sd.pop("bias")
        sd["extra"] = lucid.ones((1,))
        result = model.load_state_dict(sd, strict=False)
        assert result.missing_keys == ["bias"]
        assert result.unexpected_keys == ["extra"]

    def test_load_state_dict_preserves_target_dtype(self):
        src = nn.Linear(4, 2)
        dst = nn.Linear(4, 2).double()
        dst.load_state_dict(src.state_dict())
        for p in dst.parameters():
            assert p.dtype is lucid.float64

    def test_load_state_dict_materializes_lazy_linear(self):
        src = nn.Linear(4, 2)
        dst = nn.LazyLinear(2)
        result = dst.load_state_dict(src.state_dict())
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert dst.in_features == 4
        assert dst.weight.shape == (2, 4)
        assert dst.bias.shape == (2,)
        x = make_tensor((3, 4))
        assert dst(x).shape == (3, 2)

    def test_load_state_dict_materializes_nested_lazy_linear(self):
        src = nn.Sequential(nn.Linear(4, 2))
        dst = nn.Sequential(nn.LazyLinear(2))
        dst.load_state_dict(src.state_dict())
        assert dst[0].in_features == 4
        assert dst(make_tensor((3, 4))).shape == (3, 2)

    def test_extra_state_roundtrip_for_nested_module(self):
        src = nn.Sequential(ExtraStateModule())
        src[0].value = 42
        dst = nn.Sequential(ExtraStateModule())
        dst.load_state_dict(src.state_dict())
        assert dst[0].value == 42


class TestTrainEval:
    def test_default_is_training(self):
        model = SimpleMLP()
        assert model.training

    def test_eval_sets_training_false(self):
        model = SimpleMLP()
        model.eval()
        assert not model.training
        for m in model.modules():
            assert not m.training

    def test_train_restores_training(self):
        model = SimpleMLP()
        model.eval()
        model.train()
        assert model.training


class TestHooks:
    def test_forward_hook_called(self):
        model = nn.Linear(4, 2)
        called = []

        def hook(module, inp, out):
            called.append(True)

        handle = model.register_forward_hook(hook)
        x = make_tensor((3, 4))
        model(x)
        assert len(called) == 1
        handle.remove()

    def test_hook_removed(self):
        model = nn.Linear(4, 2)
        called = []

        def hook(module, inp, out):
            called.append(True)

        handle = model.register_forward_hook(hook)
        handle.remove()
        x = make_tensor((3, 4))
        model(x)
        assert len(called) == 0

    def test_forward_pre_hook_with_kwargs_can_modify_call(self):
        class Scale(nn.Module):
            def forward(self, x, scale=1.0):
                return x * scale

        model = Scale()

        def hook(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["scale"] = 3.0
            return args, kwargs

        handle = model.register_forward_pre_hook(hook, with_kwargs=True)
        out = model(lucid.ones((1,)), scale=1.0)
        assert out.item() == 3.0
        handle.remove()

    def test_forward_hook_with_kwargs_can_replace_output(self):
        class Scale(nn.Module):
            def forward(self, x, scale=1.0):
                return x * scale

        model = Scale()

        def hook(module, args, kwargs, output):
            return output + kwargs["scale"]

        handle = model.register_forward_hook(hook, with_kwargs=True)
        out = model(lucid.ones((1,)), scale=2.0)
        assert out.item() == 4.0
        handle.remove()

    def test_forward_hooks_prepend_order(self):
        model = nn.Identity()
        order = []

        def first(module, args, output):
            order.append("first")

        def second(module, args, output):
            order.append("second")

        h1 = model.register_forward_hook(first)
        h2 = model.register_forward_hook(second, prepend=True)
        model(lucid.ones((1,)))
        assert order == ["second", "first"]
        h1.remove()
        h2.remove()

    def test_global_forward_hooks(self):
        called = []

        def hook(module, args, output):
            called.append(type(module).__name__)

        handle = nn.register_module_forward_hook(hook)
        nn.Identity()(lucid.ones((1,)))
        assert called == ["Identity"]
        handle.remove()

    def test_forward_hook_always_call_on_exception(self):
        class Boom(nn.Module):
            def forward(self, x):
                raise RuntimeError("boom")

        called = []

        def hook(module, args, output):
            called.append(output)

        model = Boom()
        handle = model.register_forward_hook(hook, always_call=True)
        with pytest.raises(RuntimeError, match="boom"):
            model(lucid.ones((1,)))
        assert called == [None]
        handle.remove()

    def test_full_backward_hook_called(self):
        model = nn.Linear(2, 1)
        called = []

        def hook(module, grad_input, grad_output):
            called.append((len(grad_input), len(grad_output)))

        handle = model.register_full_backward_hook(hook)
        x = make_tensor((3, 2), requires_grad=True)
        loss = model(x).sum()
        loss.backward()
        assert called == [(1, 1)]
        handle.remove()

    def test_full_backward_pre_hook_can_scale_grad(self):
        model = nn.Identity()

        def hook(module, grad_output):
            return (grad_output[0] * 2.0,)

        handle = model.register_full_backward_pre_hook(hook)
        x = make_tensor((1,), requires_grad=True)
        y = model(x)
        y.sum().backward()
        assert x.grad.item() == 2.0
        handle.remove()

    def test_backward_hooks_prepend_order(self):
        model = nn.Identity()
        order = []

        def first(module, grad_input, grad_output):
            order.append("first")

        def second(module, grad_input, grad_output):
            order.append("second")

        h1 = model.register_full_backward_hook(first)
        h2 = model.register_full_backward_hook(second, prepend=True)
        x = make_tensor((1,), requires_grad=True)
        model(x).sum().backward()
        assert order == ["second", "first"]
        h1.remove()
        h2.remove()

    def test_global_backward_hooks(self):
        called = []

        def hook(module, grad_input, grad_output):
            called.append(type(module).__name__)

        handle = nn.register_module_full_backward_hook(hook)
        x = make_tensor((1,), requires_grad=True)
        nn.Identity()(x).sum().backward()
        assert called == ["Identity"]
        handle.remove()

    def test_global_backward_pre_hook_can_scale_grad(self):
        def hook(module, grad_output):
            return (grad_output[0] * 3.0,)

        handle = nn.register_module_full_backward_pre_hook(hook)
        x = make_tensor((1,), requires_grad=True)
        nn.Identity()(x).sum().backward()
        assert x.grad.item() == 3.0
        handle.remove()

    def test_removed_forward_hook_cleans_metadata(self):
        model = nn.Identity()
        called = []

        def hook(module, args, kwargs, output):
            called.append(True)

        handle = model.register_forward_hook(hook, with_kwargs=True, always_call=True)
        key = next(iter(model._forward_hooks))
        handle.remove()
        assert key not in model._forward_hooks_with_kwargs
        assert key not in model._forward_hooks_always_called
        model(lucid.ones((1,)))
        assert called == []

    def test_backward_hook_attaches_to_tuple_outputs(self):
        class Pair(nn.Module):
            def forward(self, x):
                return x, x * 2.0

        called = []

        def hook(module, grad_input, grad_output):
            called.append((len(grad_input), len(grad_output)))

        model = Pair()
        handle = model.register_full_backward_hook(hook)
        x = make_tensor((1,), requires_grad=True)
        a, b = model(x)
        (a + b).sum().backward()
        assert called == [(1, 2)]
        handle.remove()

    def test_backward_hook_sees_all_input_grads_once(self):
        class PairMul(nn.Module):
            def forward(self, x, y):
                return (x * y).sum()

        model = PairMul()
        calls = []

        def hook(module, grad_input, grad_output):
            calls.append((len(grad_input), len(grad_output)))
            return grad_input[0] * 3.0, grad_input[1] * 5.0

        handle = model.register_full_backward_hook(hook)
        x = lucid.tensor([2.0, 3.0], requires_grad=True)
        y = lucid.tensor([5.0, 7.0], requires_grad=True)
        model(x, y).backward()

        assert calls == [(2, 1)]
        np.testing.assert_allclose(x.grad.numpy(), np.array([15.0, 21.0]))
        np.testing.assert_allclose(y.grad.numpy(), np.array([10.0, 15.0]))
        handle.remove()

    def test_backward_pre_hook_scales_all_input_grads(self):
        class PairMul(nn.Module):
            def forward(self, x, y):
                return (x * y).sum()

        model = PairMul()

        def hook(module, grad_output):
            return (grad_output[0] * 2.0,)

        handle = model.register_full_backward_pre_hook(hook)
        x = lucid.tensor([2.0, 3.0], requires_grad=True)
        y = lucid.tensor([5.0, 7.0], requires_grad=True)
        model(x, y).backward()

        np.testing.assert_allclose(x.grad.numpy(), np.array([10.0, 14.0]))
        np.testing.assert_allclose(y.grad.numpy(), np.array([4.0, 6.0]))
        handle.remove()


class TestModuleContainers:
    def test_sequential(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        x = make_tensor((3, 8))
        out = model(x)
        assert out.shape == (3, 4)

    def test_module_list(self):
        layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        x = make_tensor((2, 4))
        for l in layers:
            x = l(x)
        assert x.shape == (2, 4)

    def test_module_list_extend_insert_slice_delete(self):
        layers = nn.ModuleList([nn.Linear(4, 4)])
        layers.extend([nn.ReLU(), nn.Identity()])
        layers.insert(1, nn.Tanh())
        assert len(layers) == 4
        assert isinstance(layers[1:3], nn.ModuleList)
        del layers[1]
        assert len(layers) == 3

    def test_module_dict(self):
        mdict = nn.ModuleDict({"l1": nn.Linear(4, 8), "l2": nn.Linear(8, 4)})
        x = make_tensor((2, 4))
        x = mdict["l1"](x)
        x = mdict["l2"](x)
        assert x.shape == (2, 4)

    def test_module_dict_mapping_methods(self):
        mdict = nn.ModuleDict({"a": nn.Identity()})
        mdict.update({"b": nn.ReLU()})
        assert list(mdict.keys()) == ["a", "b"]
        assert "a" in mdict
        popped = mdict.pop("a")
        assert isinstance(popped, nn.Identity)
        mdict.clear()
        assert len(mdict) == 0

    def test_parameter_containers(self):
        plist = nn.ParameterList([nn.Parameter(lucid.ones((1,)))])
        plist.extend([nn.Parameter(lucid.zeros((1,)))])
        assert len(list(plist)) == 2
        del plist[0]
        assert len(plist) == 1

        pdict = nn.ParameterDict({"w": nn.Parameter(lucid.ones((1,)))})
        pdict.update({"b": nn.Parameter(lucid.zeros((1,)))})
        assert list(pdict.keys()) == ["w", "b"]
        assert isinstance(pdict.pop("w"), nn.Parameter)
