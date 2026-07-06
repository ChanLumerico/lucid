"""``lucid.quantization`` Phase-3 — dynamic PTQ (no calibration)."""

import contextlib
from collections.abc import Iterator

import numpy as np

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized.dynamic as nnqd
import lucid.quantization as Q


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    """Pin the dequant (reference) path so structure asserts don't see MLX routing."""
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


def _linear_stack() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)
    )


class TestDynamicLinear:
    def test_swaps_without_calibration(self) -> None:
        with _reference_engine():
            qm = Q.quantize_dynamic(_linear_stack())
        assert isinstance(qm[0], nnqd.Linear)
        assert qm[0].weight_int8.dtype is lucid.int8

    def test_error_bounded(self) -> None:
        m = _linear_stack()
        m.eval()
        qm = Q.quantize_dynamic(m)
        x = lucid.randn(16, 32)
        yf, yq = m(x).numpy(), qm(x).numpy()
        assert np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9) < 0.05

    def test_qconfig_spec_filters(self) -> None:
        # Empty spec → nothing quantized.
        qm = Q.quantize_dynamic(_linear_stack(), qconfig_spec=set())
        assert not isinstance(qm[0], nnqd.Linear)


class TestDynamicLSTM:
    def test_int8_weights_and_clean_state_dict(self) -> None:
        lucid.manual_seed(1)
        lstm = nn.LSTM(8, 16, batch_first=True)
        lstm.eval()
        qm = Q.quantize_dynamic(lstm)
        assert isinstance(qm, nnqd.LSTM)
        assert qm.weight_ih_l0_int8.dtype is lucid.int8
        # state_dict holds the int8 form only — no float weight leaks from the shell.
        keys = [k for k in qm.state_dict() if k.startswith("weight_ih_l0")]
        assert keys == ["weight_ih_l0_int8", "weight_ih_l0_scale"]

    def test_output_close_to_float(self) -> None:
        lucid.manual_seed(2)
        lstm = nn.LSTM(8, 16, batch_first=True)
        lstm.eval()
        qm = Q.quantize_dynamic(lstm)
        xs = lucid.randn(4, 5, 8)
        yf = lstm(xs)[0].numpy()
        yq = qm(xs)[0].numpy()
        assert np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9) < 0.03
