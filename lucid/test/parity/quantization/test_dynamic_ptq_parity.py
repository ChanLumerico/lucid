"""Parity: dynamic PTQ quality vs the reference framework.

Dynamic-quant *numerics* differ by kernel (weight scheme, activation grid),
so this is a **quality** parity: given identical float weights, lucid's
dynamic quantization must stay as close to the float output as the
reference's dynamic quantization does.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.quantization as Q


@pytest.mark.parity
class TestDynamicLinearParity:
    def test_quality_matches_reference(self, ref: Any) -> None:
        rng = np.random.default_rng(0)
        w = rng.standard_normal((16, 8)).astype(np.float32)
        b = rng.standard_normal(16).astype(np.float32)
        x = rng.standard_normal((8, 8)).astype(np.float32)
        y_float = x @ w.T + b

        ll = nn.Linear(8, 16)
        ll.weight = nn.Parameter(lucid.tensor(w.copy()))
        ll.bias = nn.Parameter(lucid.tensor(b.copy()))
        ll.eval()
        lq = Q.quantize_dynamic(ll)
        y_lucid = lq(lucid.tensor(x.copy())).numpy()

        rl = ref.nn.Linear(8, 16)
        with ref.no_grad():
            rl.weight.copy_(ref.tensor(w.copy()))
            rl.bias.copy_(ref.tensor(b.copy()))
        rl.eval()
        rq = ref.quantization.quantize_dynamic(rl, {ref.nn.Linear}, dtype=ref.qint8)
        y_ref = rq(ref.tensor(x.copy())).detach().numpy()

        err_lucid = np.abs(y_lucid - y_float).mean() / (np.abs(y_float).mean() + 1e-9)
        err_ref = np.abs(y_ref - y_float).mean() / (np.abs(y_float).mean() + 1e-9)
        assert err_lucid < 0.05  # lucid dynamic quant is accurate
        assert abs(err_lucid - err_ref) < 0.05  # ...and comparable to the reference
