"""
Tests for lucid.save / lucid.load.
"""

import pytest
import tempfile
import os
import numpy as np
import lucid
import lucid.nn as nn
from conftest import assert_close


def _tmp_path():
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    f.close()
    return f.name


class TestSaveLoad:
    def test_tensor_round_trip(self):
        path = _tmp_path()
        try:
            x = lucid.randn(3, 4)
            lucid.save(x, path)
            loaded = lucid.load(path)
            assert_close(loaded.numpy(), x.numpy())
        finally:
            os.unlink(path)

    def test_dict_round_trip(self):
        path = _tmp_path()
        try:
            data = {"x": lucid.randn(2, 3), "meta": {"lr": 0.01, "epoch": 5}}
            lucid.save(data, path)
            loaded = lucid.load(path)
            assert_close(loaded["x"].numpy(), data["x"].numpy())
            assert loaded["meta"]["lr"] == 0.01
        finally:
            os.unlink(path)

    def test_weights_only_default(self):
        path = _tmp_path()
        try:
            x = lucid.randn(2)
            lucid.save(x, path)
            loaded = lucid.load(path)  # weights_only=True by default
            assert_close(loaded.numpy(), x.numpy())
        finally:
            os.unlink(path)

    def test_map_location_string(self):
        path = _tmp_path()
        try:
            x = lucid.randn(2, 3)
            lucid.save(x, path)
            loaded = lucid.load(path, map_location="cpu")
            assert not loaded.is_metal
        finally:
            os.unlink(path)

    def test_map_location_dict(self):
        path = _tmp_path()
        try:
            x = lucid.randn(2, 3)
            lucid.save(x, path)
            loaded = lucid.load(path, map_location={"cpu": "cpu", "metal": "cpu"})
            assert not loaded.is_metal
        finally:
            os.unlink(path)

    def test_map_location_callable(self):
        path = _tmp_path()
        try:
            x = lucid.randn(2)
            lucid.save(x, path)
            loaded = lucid.load(path, map_location=lambda t, loc: t.cpu())
            assert not loaded.is_metal
        finally:
            os.unlink(path)

    def test_invalid_format_raises(self):
        path = _tmp_path()
        try:
            with open(path, "wb") as f:
                f.write(b"not a lucid checkpoint")
            with pytest.raises((RuntimeError, Exception)):
                lucid.load(path)
        finally:
            os.unlink(path)

    def test_model_state_dict_round_trip(self):
        path = _tmp_path()
        try:
            model = nn.Linear(4, 2)
            sd = model.state_dict()
            lucid.save(sd, path)
            loaded_sd = lucid.load(path)
            model2 = nn.Linear(4, 2)
            model2.load_state_dict(loaded_sd)
            assert_close(model.weight.numpy(), model2.weight.numpy())
        finally:
            os.unlink(path)
