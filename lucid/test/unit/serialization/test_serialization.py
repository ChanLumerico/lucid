"""``lucid.save`` / ``lucid.load`` round-trip + Module state_dict."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestSaveLoadTensor:
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.lucid"
            t = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
            lucid.save(t, str(path))
            loaded = lucid.load(str(path))
            np.testing.assert_array_equal(loaded.numpy(), t.numpy())

    def test_dtype_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.lucid"
            t = lucid.tensor([1, 2, 3], dtype=lucid.int64)
            lucid.save(t, str(path))
            loaded = lucid.load(str(path))
            assert loaded.dtype == lucid.int64


class TestSaveLoadStateDict:
    def test_module_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.lucid"
            model = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )
            sd = model.state_dict()
            lucid.save(sd, str(path))

            model2 = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )
            loaded = lucid.load(str(path), weights_only=False)
            model2.load_state_dict(loaded)

            # Compare any one parameter — they must now be equal.
            for k, v in sd.items():
                v2 = dict(model2.state_dict())[k]
                np.testing.assert_allclose(
                    v.numpy(),
                    v2.numpy(),
                    atol=1e-6,
                    err_msg=f"parameter {k} drifted on round-trip",
                )


class TestSaveLoadDict:
    def test_dict_of_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.lucid"
            d = {"a": lucid.tensor([1.0]), "b": lucid.tensor([2.0, 3.0])}
            lucid.save(d, str(path))
            loaded = lucid.load(str(path), weights_only=False)
            np.testing.assert_array_equal(loaded["a"].numpy(), [1.0])
            np.testing.assert_array_equal(loaded["b"].numpy(), [2.0, 3.0])
