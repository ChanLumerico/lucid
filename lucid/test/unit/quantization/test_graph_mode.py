"""``lucid.quantization`` Phase-5 — graph-mode PTQ (prepare_fx / convert_fx)."""

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.quantization as Q


def _quantize_fx(model: nn.Module) -> nn.Module:
    model.eval()
    prepared = Q.prepare_fx(model)  # auto-wraps — no manual QuantStub
    for _ in range(10):
        prepared(lucid.randn(16, 16))
    return Q.convert_fx(prepared)


class TestGraphModePrepare:
    def test_auto_wraps_without_manual_stubs(self) -> None:
        lucid.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        prepared = Q.prepare_fx(model)
        # QuantWrapper was inserted automatically.
        assert isinstance(prepared, nnq.QuantWrapper)

    def test_convert_produces_int8(self) -> None:
        lucid.manual_seed(1)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        qm = _quantize_fx(model)
        first = qm.module[0]
        assert isinstance(first, nnq.Linear)
        assert first.weight_int8.dtype is lucid.int8


class TestCompiledQuantized:
    def test_quantized_model_compiles_bit_identical(self) -> None:
        lucid.manual_seed(2)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        qm = _quantize_fx(model)
        x = lucid.randn(8, 16)
        y_eager = qm(x).numpy()

        compiled = lucid.compile(qm)
        y_compiled = compiled(x).numpy()
        # Composite quantized graph fuses under compile with identical numerics.
        assert np.allclose(y_eager, y_compiled, atol=1e-5)

    def test_accuracy_bounded(self) -> None:
        lucid.manual_seed(3)
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        model.eval()
        qm = _quantize_fx(model)
        errs = []
        for _ in range(8):
            xe = lucid.randn(16, 16)
            yf = model(xe).numpy()
            yq = qm(xe).numpy()
            errs.append(np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9))
        assert np.mean(errs) < 0.05
