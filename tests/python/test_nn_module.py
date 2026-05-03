"""
Tests for nn.Module, Parameter, state_dict, hooks.
"""

import pytest
import numpy as np
import lucid
import lucid.nn as nn
from conftest import assert_close


class TestModule:
    def test_forward_raises_not_implemented(self):
        m = nn.Module()
        with pytest.raises(NotImplementedError):
            m(lucid.zeros(1))

    def test_parameters(self):
        fc = nn.Linear(4, 2)
        params = list(fc.parameters())
        assert len(params) == 2  # weight + bias

    def test_named_parameters(self):
        fc = nn.Linear(4, 2)
        names = [n for n, _ in fc.named_parameters()]
        assert "weight" in names
        assert "bias" in names

    def test_children(self):
        seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        children = list(seq.children())
        assert len(children) == 2

    def test_train_eval(self):
        fc = nn.Linear(4, 2)
        assert fc.training
        fc.eval()
        assert not fc.training
        fc.train()
        assert fc.training

    def test_train_invalid_type(self):
        fc = nn.Linear(4, 2)
        with pytest.raises(TypeError):
            fc.train(1)  # type: ignore[arg-type]

    def test_zero_grad(self):
        fc = nn.Linear(4, 2)
        x = lucid.randn(3, 4)
        fc(x).mean().backward()
        assert fc.weight.grad is not None
        fc.zero_grad()
        assert fc.weight.grad is None

    def test_get_submodule(self):
        seq = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        m = seq.get_submodule("0")
        assert isinstance(m, nn.Linear)

    def test_get_submodule_nested(self):
        model = nn.Sequential(
            nn.Sequential(nn.Linear(4, 2))
        )
        m = model.get_submodule("0.0")
        assert isinstance(m, nn.Linear)

    def test_get_submodule_missing(self):
        seq = nn.Sequential()
        with pytest.raises(AttributeError):
            seq.get_submodule("nonexistent")

    def test_get_parameter(self):
        fc = nn.Linear(4, 2)
        p = fc.get_parameter("weight")
        assert p is fc.weight

    def test_repr_with_extra_repr(self):
        fc = nn.Linear(4, 2)
        r = repr(fc)
        assert "in_features=4" in r
        assert "out_features=2" in r

    def test_repr_sequential(self):
        seq = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
        r = repr(seq)
        assert "(0): Linear" in r
        assert "(1): ReLU" in r

    def test_repr_leaf_no_children(self):
        relu = nn.ReLU()
        assert repr(relu) == "ReLU()"


class TestHooks:
    def test_forward_pre_hook(self):
        calls = []
        fc = nn.Linear(4, 2)
        h = fc.register_forward_pre_hook(lambda m, args: calls.append("pre"))
        fc(lucid.randn(2, 4))
        assert calls == ["pre"]
        h.remove()
        fc(lucid.randn(2, 4))
        assert calls == ["pre"]  # not called again after remove

    def test_forward_hook(self):
        calls = []
        fc = nn.Linear(4, 2)
        h = fc.register_forward_hook(lambda m, args, out: calls.append("post"))
        fc(lucid.randn(2, 4))
        assert calls == ["post"]
        h.remove()

    def test_pre_and_post_hooks_order(self):
        order = []
        fc = nn.Linear(4, 2)
        fc.register_forward_pre_hook(lambda m, a: order.append(1))
        fc.register_forward_hook(lambda m, a, o: order.append(2))
        fc(lucid.randn(2, 4))
        assert order == [1, 2]


class TestStateDict:
    def test_save_load_round_trip(self):
        fc1 = nn.Linear(4, 2)
        fc2 = nn.Linear(4, 2)
        sd = fc1.state_dict()
        missing, unexpected = fc2.load_state_dict(sd)
        assert missing == [] and unexpected == []
        assert_close(fc1.weight.numpy(), fc2.weight.numpy())

    def test_strict_false(self):
        fc = nn.Linear(4, 2)
        sd = fc.state_dict()
        sd["extra_key"] = lucid.zeros(1)
        missing, unexpected = fc.load_state_dict(sd, strict=False)
        assert "extra_key" in unexpected

    def test_strict_true_raises(self):
        fc = nn.Linear(4, 2)
        sd = fc.state_dict()
        sd["extra_key"] = lucid.zeros(1)
        with pytest.raises(RuntimeError):
            fc.load_state_dict(sd, strict=True)

    def test_parameter_identity_preserved_after_load(self):
        fc = nn.Linear(4, 2)
        weight_id = id(fc.weight)
        sd = fc.state_dict()
        fc.load_state_dict(sd)
        assert id(fc.weight) == weight_id  # same Python object

    def test_to_preserves_identity(self):
        fc = nn.Linear(4, 2)
        weight_id = id(fc.weight)
        fc.to("cpu")
        assert id(fc.weight) == weight_id
