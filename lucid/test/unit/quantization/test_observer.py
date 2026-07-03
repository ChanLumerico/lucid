"""``lucid.quantization`` Phase-1 — observers, FakeQuantize, QConfig."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.quantization as Q


class TestMinMaxObserver:
    def test_running_min_max(self) -> None:
        obs = Q.MinMaxObserver()
        obs(lucid.tensor([[-1.0, 2.0], [0.5, 3.0]]))
        obs(lucid.tensor([[-2.0, 1.0]]))
        assert obs.min_val.item() == -2.0
        assert obs.max_val.item() == 3.0

    def test_identity_passthrough(self) -> None:
        obs = Q.MinMaxObserver()
        x = lucid.tensor([1.0, 2.0, 3.0])
        out = obs(x)
        assert np.array_equal(out.numpy(), x.numpy())

    def test_qparams_affine_quint8(self) -> None:
        obs = Q.MinMaxObserver(qscheme=Q.per_tensor_affine, qdtype=Q.quint8)
        obs(lucid.tensor([-1.0, 3.0]))
        scale, zp = obs.calculate_qparams()
        assert scale.item() == pytest.approx(4.0 / 255.0, rel=1e-5)
        assert 0.0 <= zp.item() <= 255.0


class TestMovingAverageObserver:
    def test_ema_smooths(self) -> None:
        obs = Q.MovingAverageMinMaxObserver(averaging_constant=0.5)
        obs(lucid.tensor([-2.0, 2.0]))  # seed
        obs(lucid.tensor([-4.0, 4.0]))  # EMA halfway toward -4/4
        assert obs.min_val.item() == pytest.approx(-3.0)
        assert obs.max_val.item() == pytest.approx(3.0)


class TestPerChannelObserver:
    def test_per_channel_stats(self) -> None:
        obs = Q.PerChannelMinMaxObserver(ch_axis=0)
        obs(lucid.tensor([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.1]]))
        assert obs.min_val.numpy().tolist() == [1.0, -4.0]
        assert obs.max_val.numpy().tolist() == [3.0, 0.5]
        scale, zp = obs.calculate_qparams()
        assert scale.shape == (2,)
        assert zp.numpy().tolist() == [0.0, 0.0]  # symmetric


class TestHistogramObserver:
    def test_collects_and_valid_qparams(self) -> None:
        import random

        random.seed(0)
        obs = Q.HistogramObserver(bins=256)
        obs(lucid.tensor([random.gauss(0.0, 1.0) for _ in range(2000)]))
        assert obs.histogram.numpy().sum() == pytest.approx(2000, abs=2)
        scale, zp = obs.calculate_qparams()
        assert scale.item() > 0.0

    def test_clips_heavy_tail(self) -> None:
        import random

        random.seed(1)
        data = [random.gauss(0.0, 1.0) for _ in range(3000)] + [
            random.uniform(5.0, 12.0) for _ in range(150)
        ]
        h = Q.HistogramObserver(bins=512)
        h(lucid.tensor(data))
        m = Q.MinMaxObserver()
        m(lucid.tensor(data))
        # L2 clip range is no wider than raw min/max.
        assert h.calculate_qparams()[0].item() <= m.calculate_qparams()[0].item()


class TestFakeQuantize:
    def test_observe_and_toggle(self) -> None:
        fq = Q.FakeQuantize(observer=Q.MinMaxObserver)
        x = lucid.tensor([-1.0, 0.0, 2.0])
        fq(x)  # observes + fake-quants
        assert fq.scale.item() > 0.0
        # Freeze observer: scale stays put even on a wider batch.
        frozen = fq.scale.item()
        fq.disable_observer()
        fq(lucid.tensor([-100.0, 100.0]))
        assert fq.scale.item() == frozen

    def test_disable_fake_quant_is_identity(self) -> None:
        fq = Q.FakeQuantize(observer=Q.MinMaxObserver).disable_fake_quant()
        x = lucid.tensor([0.123, -4.5, 7.8])
        assert np.array_equal(fq(x).numpy(), x.numpy())

    def test_ste_gradient(self) -> None:
        fq = Q.FakeQuantize(observer=Q.MinMaxObserver)
        x = lucid.tensor([-1.0, 0.2, 0.5, 2.0], requires_grad=True)
        fq(x).sum().backward()
        assert x.grad is not None


class TestQConfig:
    def test_defaults_instantiate(self) -> None:
        qc = Q.get_default_qconfig()
        assert isinstance(qc.weight(), Q.PerChannelMinMaxObserver)
        assert isinstance(qc.activation(), Q.HistogramObserver)
        qat = Q.get_default_qat_qconfig()
        assert isinstance(qat.weight(), Q.FakeQuantize)

    def test_mapping_resolution_order(self) -> None:
        qc, qat = Q.get_default_qconfig(), Q.get_default_qat_qconfig()
        m = (
            Q.QConfigMapping()
            .set_global(qc)
            .set_object_type(nn.Linear, qat)
            .set_module_name("head", None)
        )
        assert m.get_qconfig(nn.Conv2d, "backbone") is qc  # global
        assert m.get_qconfig(nn.Linear, "fc") is qat  # by type
        assert m.get_qconfig(nn.Linear, "head") is None  # by name wins over type


class TestObserverStateDict:
    def test_round_trip(self) -> None:
        obs = Q.MinMaxObserver()
        obs(lucid.tensor([-2.0, 5.0]))
        sd = obs.state_dict()
        assert "min_val" in sd and "max_val" in sd
        fresh = Q.MinMaxObserver()
        fresh.load_state_dict(sd)
        assert fresh.min_val.item() == -2.0
        assert fresh.max_val.item() == 5.0
