"""``lucid.quantization`` Phase-6 — real int4/int8 GEMM via MLX (Metal only)."""

import numpy as np
import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:
        return False


_ENABLED = backends.quantized.available and _metal_ok()

pytestmark = pytest.mark.skipif(
    not _ENABLED, reason="engine lacks quantized ops or Metal unavailable"
)


class TestRealGEMM:
    def test_int8_accuracy(self) -> None:
        lucid.manual_seed(0)
        lin = nn.Linear(256, 256)
        lin.eval()
        x = lucid.randn(16, 256)
        yf = lin(x).numpy()
        q = nnq.QuantizedLinearMLX.from_float(lin, bits=8)
        yq = q(x.to("metal")).numpy()
        assert np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9) < 0.02

    def test_int4_accuracy(self) -> None:
        lucid.manual_seed(1)
        lin = nn.Linear(256, 256)
        lin.eval()
        x = lucid.randn(16, 256)
        yf = lin(x).numpy()
        q = nnq.QuantizedLinearMLX.from_float(lin, bits=4)
        yq = q(x.to("metal")).numpy()
        # 4-bit is coarser but still tracks the float output.
        assert np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9) < 0.15

    def test_weight_memory_smaller(self) -> None:
        lucid.manual_seed(2)
        lin = nn.Linear(512, 512, bias=False)
        q = nnq.QuantizedLinearMLX.from_float(lin, bits=8)
        float_bytes = 512 * 512 * 4
        packed_bytes = (
            q.packed_weight.numpy().nbytes
            + q.scales.numpy().nbytes
            + q.biases.numpy().nbytes
        )
        assert packed_bytes < float_bytes / 2  # meaningfully smaller

    def test_engine_op_matches_dequant_path(self) -> None:
        # The real GEMM should agree with the reference dequant→float path.
        from lucid.quantization import _qgemm

        lucid.manual_seed(3)
        w = lucid.randn(64, 128).to("metal")
        x = lucid.randn(8, 128).to("metal")
        packed, scales, biases = _qgemm.quantize(w, group_size=64, bits=8)
        y_kernel = _qgemm.quantized_matmul(
            x, packed, scales, biases, transpose=True, group_size=64, bits=8
        ).numpy()
        w_deq = _qgemm.dequantize(packed, scales, biases, group_size=64, bits=8)
        y_deq = lucid.matmul(x, w_deq.mT).numpy()
        assert np.allclose(y_kernel, y_deq, atol=1e-3)


class TestBackendsToggle:
    def test_toggle_values(self) -> None:
        assert backends.quantized.engine == "auto"
        backends.quantized.engine = "mlx_group"
        assert backends.quantized.use_mlx()
        backends.quantized.engine = "reference"
        assert not backends.quantized.use_mlx()
        backends.quantized.engine = "auto"  # restore
        with pytest.raises(ValueError):
            backends.quantized.engine = "nonsense"
