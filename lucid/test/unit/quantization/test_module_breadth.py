"""``lucid.quantization`` Phase-7 — quantized ConvTranspose + activations."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.quantization as Q


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


_CT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
_QCT = {1: nnq.ConvTranspose1d, 2: nnq.ConvTranspose2d, 3: nnq.ConvTranspose3d}
_CT_SHAPE = {1: (2, 8, 8), 2: (2, 8, 8, 8), 3: (2, 8, 4, 4, 4)}


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_conv_transpose_quantizes(rank: int) -> None:
    with _reference_engine():
        lucid.manual_seed(rank)
        m = nn.Sequential(_CT[rank](8, 4, 3, stride=2, padding=1, output_padding=1))
        m.eval()
        x = lucid.randn(*_CT_SHAPE[rank])
        yf = m(x).numpy()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(6):
            prepared(lucid.randn(*_CT_SHAPE[rank]))
        qm = Q.convert(prepared)
        assert isinstance(qm[0], _QCT[rank])
        assert qm[0].weight_int8.dtype is lucid.int8
        yq = qm(x).numpy()
        assert yq.shape == yf.shape
        # mean relative error — the meaningful accuracy metric (per-channel int8
        # on random weights leaves the odd single-element max-error outlier).
        assert np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9) < 0.05


_ACTS = [
    (nn.Sigmoid, nnq.Sigmoid),
    (nn.Hardswish, nnq.Hardswish),
    (nn.Hardsigmoid, nnq.Hardsigmoid),
    (nn.Tanh, nnq.Tanh),
    (nn.ELU, nnq.ELU),
    (nn.LeakyReLU, nnq.LeakyReLU),
]


@pytest.mark.parametrize("float_cls,q_cls", _ACTS)
def test_activation_quantizes(float_cls: type, q_cls: type) -> None:
    with _reference_engine():
        lucid.manual_seed(0)
        m = nn.Sequential(nn.Linear(64, 64), float_cls())
        m.eval()
        x = lucid.randn(8, 64)
        yf = m(x).numpy()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(5):
            prepared(lucid.randn(8, 64))
        qm = Q.convert(prepared)
        assert isinstance(qm[1], q_cls)
        yq = qm(x).numpy()
        assert np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9) < 0.1


def test_leaky_relu_carries_slope() -> None:
    with _reference_engine():
        m = nn.Sequential(nn.Linear(8, 8), nn.LeakyReLU(negative_slope=0.2))
        m.eval()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(3):
            prepared(lucid.randn(4, 8))
        qm = Q.convert(prepared)
        assert isinstance(qm[1], nnq.LeakyReLU)
        assert qm[1].negative_slope == pytest.approx(0.2)
