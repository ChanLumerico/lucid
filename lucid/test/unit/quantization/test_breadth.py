"""``lucid.quantization`` Phase-7 — breadth (Embedding) + robustness."""

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.quantization as Q


class TestQuantizedEmbedding:
    def test_int8_table_and_accuracy(self) -> None:
        lucid.manual_seed(0)
        emb = nn.Embedding(100, 32)
        emb.eval()
        idx = lucid.tensor([[1, 5, 20], [3, 8, 50]], dtype=lucid.int64)
        yf = emb(idx).numpy()
        q = nnq.Embedding.from_float(emb)
        assert q.weight_int8.dtype is lucid.int8
        assert q.weight_int8.shape == (100, 32)
        yq = q(idx).numpy()
        assert np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9) < 0.02

    def test_table_memory_smaller(self) -> None:
        emb = nn.Embedding(256, 64)
        q = nnq.Embedding.from_float(emb)
        float_bytes = 256 * 64 * 4
        packed = q.weight_int8.numpy().nbytes + q.weight_scale.numpy().nbytes
        assert packed < float_bytes / 2

    def test_convert_maps_embedding(self) -> None:
        lucid.manual_seed(1)
        model = nn.Sequential(nn.Embedding(50, 16))
        qm = Q.convert(Q.prepare(model))  # embedding converts w/o calibration
        assert isinstance(qm[0], nnq.Embedding)


class TestIntegerInputPassthrough:
    def test_quantstub_skips_integer_input(self) -> None:
        # QuantStub / Quantize must not fake-quantize integer indices.
        stub = nnq.QuantStub()
        idx = lucid.tensor([1, 2, 3], dtype=lucid.int64)
        assert np.array_equal(stub(idx).numpy(), idx.numpy())

        quant = nnq.Quantize(lucid.tensor(0.05), lucid.tensor(0.0))
        assert np.array_equal(quant(idx).numpy(), idx.numpy())
