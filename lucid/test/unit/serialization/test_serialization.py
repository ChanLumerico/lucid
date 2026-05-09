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


class TestStateDictV2:
    """state-dict-v2: _metadata version propagation + assign= parameter."""

    def test_metadata_version_in_simple_module(self) -> None:
        import lucid.nn as nn
        m = nn.Linear(3, 2)
        sd = m.state_dict()
        assert hasattr(sd, '_metadata')
        assert sd._metadata.get('')['version'] == 1  # type: ignore[index]

    def test_metadata_propagates_to_children(self) -> None:
        import lucid.nn as nn
        seq = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 3))
        sd = seq.state_dict()
        meta = sd._metadata
        for key in ('', '0', '1', '2'):
            assert key in meta, f"'{key}' missing from _metadata"
            assert meta[key]['version'] == 1

    def test_batchnorm_metadata_version_2(self) -> None:
        import lucid.nn as nn
        bn = nn.BatchNorm2d(4)
        sd = bn.state_dict()
        assert sd._metadata['']['version'] == 2

    def test_load_state_dict_returns_incompatible_keys(self) -> None:
        import lucid.nn as nn
        import lucid
        m = nn.Linear(3, 2)
        sd = m.state_dict()
        result = m.load_state_dict(sd)
        assert result.missing_keys == []
        assert result.unexpected_keys == []

    def test_assign_false_shape_mismatch_raises(self) -> None:
        import lucid.nn as nn
        import lucid
        m = nn.Linear(3, 2)
        bad_sd = {'weight': lucid.randn(4, 3), 'bias': lucid.zeros(4)}
        with pytest.raises(RuntimeError, match="size mismatch"):
            m.load_state_dict(bad_sd, strict=True, assign=False)

    def test_assign_true_allows_shape_change(self) -> None:
        import lucid.nn as nn
        import lucid
        m = nn.Linear(3, 2)
        new_w = lucid.randn(4, 3)
        new_b = lucid.zeros(4)
        m.load_state_dict({'weight': new_w, 'bias': new_b}, strict=True, assign=True)
        assert m.weight.shape == (4, 3)
        assert m.bias.shape == (4,)

    def test_assign_true_copies_values(self) -> None:
        import lucid.nn as nn
        import lucid
        import numpy as np
        src = nn.Linear(3, 2)
        dst = nn.Linear(3, 2)
        dst.load_state_dict(src.state_dict(), assign=True)
        np.testing.assert_allclose(dst.weight.numpy(), src.weight.numpy(), atol=1e-6)

    def test_metadata_round_trip_via_save_load(self) -> None:
        """_metadata survives lucid.save / lucid.load."""
        import lucid.nn as nn
        import lucid
        import io
        m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3))
        sd = m.state_dict()
        buf = io.BytesIO()
        lucid.save(sd, buf)
        buf.seek(0)
        loaded = lucid.load(buf, weights_only=False)
        assert hasattr(loaded, '_metadata')
        assert loaded._metadata.get('')['version'] == 1  # type: ignore[index]
