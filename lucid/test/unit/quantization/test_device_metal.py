"""On-device (Metal) quantization — device-consistency regression.

Two device bugs blocked the whole static-PTQ path on Metal until 2026-07-13:

1. **Observer buffers stranded on CPU.**  Observers seed ``min_val`` / ``max_val``
   device-agnostically (``+inf`` / ``-inf`` on CPU); a per-channel *weight*
   observer's seed does not ride along on ``module.to("metal")``, so the first
   ``minimum(seed_cpu, weight_gpu)`` calibration reduction ``DeviceMismatch``-ed
   (``observer.py`` PerChannelMinMax).  Fixed: observers adopt the observed
   tensor's device on first sight (``ObserverBase._align_running_buffers``).

2. **qparams stranded on CPU.**  A HistogramObserver derives its range from host
   floats, so the activation ``scale`` / ``zero_point`` landed on CPU; ``convert``
   baked them there, and inference then ran ``round(x_gpu / scale_cpu)`` →
   ``DeviceMismatch`` in ``fake_quantize``.  Fixed: the quantize / dequantize /
   fake-quantize ops pull a tensor scale / zero-point onto the data's device.

Guards the full prepare → to(metal) → calibrate → convert → infer round-trip.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.quantization as Q


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


def _static_quant(dev: str, qconfig: object) -> lucid.Tensor:
    lucid.manual_seed(0)
    m = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8)).to(dev)
    m.eval()
    qm = Q.prepare(m, qconfig).to(dev)
    for _ in range(4):
        qm(lucid.randn(4, 16, device=dev))
    cm = Q.convert(qm)
    x = lucid.tensor([[0.1 * i - 0.8 for i in range(16)] for _ in range(4)], device=dev)
    return cm(x)


def test_static_ptq_metal_round_trip_finite() -> None:
    """prepare → to(metal) → calibrate → convert → infer must not crash / NaN."""
    y = _static_quant("metal", Q.get_default_qconfig())
    assert y.is_metal
    assert not bool((y != y).any().item())  # finite


def test_static_ptq_metal_matches_cpu() -> None:
    """The same recipe on CPU and Metal must agree up to quantization noise."""
    ym = _static_quant("metal", Q.get_default_qconfig())
    yc = _static_quant("cpu", Q.get_default_qconfig())
    assert float((ym.to("cpu") - yc).abs().max().item()) < 5e-2


def test_qat_qconfig_static_quant_on_metal() -> None:
    """The QAT (fake-quant) qconfig path also survives the Metal round-trip."""
    y = _static_quant("metal", Q.get_default_qat_qconfig())
    assert not bool((y != y).any().item())


@pytest.mark.parametrize(
    "obs_factory",
    [
        Q.MinMaxObserver,
        Q.MovingAverageMinMaxObserver,
        lambda: Q.PerChannelMinMaxObserver(ch_axis=0),
        lambda: Q.MovingAveragePerChannelMinMaxObserver(ch_axis=0),
        Q.HistogramObserver,
    ],
)
def test_observers_calibrate_on_metal(obs_factory: object) -> None:
    """Every accumulating observer must calibrate a GPU tensor without a
    DeviceMismatch and yield finite qparams."""
    obs = obs_factory()
    x = lucid.randn(4, 8, device="metal")
    obs(x)
    obs(x * 1.5)
    scale, zero_point = obs.calculate_qparams()
    assert not bool((scale != scale).any().item())
    assert not bool((zero_point != zero_point).any().item())


def test_fake_quantize_tolerates_cpu_qparams_on_metal_input() -> None:
    """The quantize ops must align a CPU scale/zero-point onto a Metal input
    (the exact HistogramObserver-derived-qparams failure mode)."""
    from lucid.quantization._functional import fake_quantize

    x = lucid.randn(4, 8, device="metal")
    scale = lucid.tensor(0.05)  # CPU
    zero_point = lucid.tensor(0.0)  # CPU
    out = fake_quantize(x, scale, zero_point, -128, 127, None)
    assert out.is_metal
    assert not bool((out != out).any().item())
