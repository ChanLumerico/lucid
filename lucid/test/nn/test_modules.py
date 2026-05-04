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
        model2.load_state_dict(sd)
        # Weights should match
        for k in sd:
            np.testing.assert_array_almost_equal(
                model2.state_dict()[k].numpy(),
                sd[k].numpy()
            )


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

    def test_module_dict(self):
        mdict = nn.ModuleDict({"l1": nn.Linear(4, 8), "l2": nn.Linear(8, 4)})
        x = make_tensor((2, 4))
        x = mdict["l1"](x)
        x = mdict["l2"](x)
        assert x.shape == (2, 4)
