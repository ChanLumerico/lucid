"""
Serialization end-to-end tests: save/load round-trip.
"""

import pytest
import tempfile
import os
import numpy as np
import lucid
import lucid.nn as nn
from lucid.test.helpers.numerics import make_tensor


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestStateDictRoundtrip:
    @pytest.mark.slow
    def test_state_dict_save_load(self):
        model = TinyNet()
        sd_before = {k: v.numpy().copy() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            lucid.save(model.state_dict(), path)
            loaded_sd = lucid.load(path)
            model2 = TinyNet()
            model2.load_state_dict(loaded_sd)
            sd_after = {k: v.numpy() for k, v in model2.state_dict().items()}

            for k in sd_before:
                np.testing.assert_array_almost_equal(
                    sd_before[k], sd_after[k], err_msg=f"Key {k} mismatch after load"
                )
        finally:
            if os.path.exists(path):
                os.remove(path)

    @pytest.mark.slow
    def test_model_outputs_same_after_load(self):
        model = TinyNet()
        x = make_tensor((3, 4))

        with (
            model.eval_mode()
            if hasattr(model, "eval_mode")
            else __import__("contextlib").nullcontext()
        ):
            out_before = model(x).numpy().copy()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        try:
            lucid.save(model.state_dict(), path)
            model2 = TinyNet()
            model2.load_state_dict(lucid.load(path))
            out_after = model2(x).numpy()
            np.testing.assert_array_almost_equal(out_before, out_after, decimal=5)
        finally:
            if os.path.exists(path):
                os.remove(path)
