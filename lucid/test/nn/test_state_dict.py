"""Tests for state_dict v2 — _load_from_state_dict, hooks, _metadata, persistent buffers."""

import os
import tempfile
from collections import OrderedDict

import pytest
import numpy as np

import lucid
import lucid.nn as nn


class TestStateDictMetadata:
    def test_state_dict_carries_metadata_attribute(self):
        bn = nn.BatchNorm2d(8)
        sd = bn.state_dict()
        assert hasattr(sd, "_metadata")
        # BatchNorm declares _version=2 → metadata at root.
        assert sd._metadata.get("", {}).get("version") == 2

    def test_state_dict_no_metadata_for_unversioned_modules(self):
        layer = nn.Linear(4, 8)
        sd = layer.state_dict()
        # Linear has no _version → entry absent (or empty).
        assert sd._metadata.get("", {}).get("version") is None

    def test_metadata_for_nested_versioned_modules(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
        sd = model.state_dict()
        # The BatchNorm at index 1 should have version metadata.
        assert sd._metadata.get("1", {}).get("version") == 2


class TestPersistentBuffers:
    def test_non_persistent_buffer_excluded_from_state_dict(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("persistent", lucid.ones(3))
                self.register_buffer("ephemeral", lucid.zeros(3), persistent=False)

        m = M()
        sd = m.state_dict()
        assert "persistent" in sd
        assert "ephemeral" not in sd

    def test_loading_does_not_require_non_persistent_buffer(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("persistent", lucid.ones(3))
                self.register_buffer("ephemeral", lucid.zeros(3), persistent=False)

        src = M()
        dst = M()
        dst.load_state_dict(src.state_dict())  # no missing-key error


class TestLoadStateDictHooks:
    def test_pre_hook_can_rename_keys(self):
        # Old checkpoint uses 'fc.weight'; new model expects 'linear.weight'.
        layer = nn.Linear(4, 8)

        def rename_hook(module, state_dict, prefix, local_metadata, strict,
                        missing, unexpected, error_msgs):
            old = f"{prefix}fc.weight"
            new = f"{prefix}weight"
            if old in state_dict and new not in state_dict:
                state_dict[new] = state_dict.pop(old)

        handle = layer.register_load_state_dict_pre_hook(rename_hook)
        try:
            sd_old = OrderedDict()
            sd_old["fc.weight"] = lucid.zeros(8, 4)
            sd_old["bias"] = lucid.zeros(8)
            sd_old._metadata = {}
            result = layer.load_state_dict(sd_old)
            assert result.missing_keys == []
            assert result.unexpected_keys == []
        finally:
            handle.remove()

    def test_post_hook_receives_incompatible_keys(self):
        layer = nn.Linear(4, 8)
        captured = []

        def post(module, incompatible_keys):
            captured.append(incompatible_keys)

        handle = layer.register_load_state_dict_post_hook(post)
        try:
            sd = layer.state_dict()
            layer.load_state_dict(sd)
            assert len(captured) == 1
            assert captured[0].missing_keys == []
            assert captured[0].unexpected_keys == []
        finally:
            handle.remove()

    def test_global_pre_hook_fires_for_every_module(self):
        from lucid.nn.hooks import register_module_load_state_dict_pre_hook

        seen: list[str] = []

        def hook(module, state_dict, prefix, local_metadata, strict,
                 missing, unexpected, error_msgs):
            seen.append(type(module).__name__)

        handle = register_module_load_state_dict_pre_hook(hook)
        try:
            model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
            model.load_state_dict(model.state_dict())
            assert "Sequential" in seen
            assert seen.count("Linear") == 2
            assert "ReLU" in seen
        finally:
            handle.remove()

    def test_handle_remove_deregisters_hook(self):
        layer = nn.Linear(4, 8)
        called = []

        def hook(*args):
            called.append(True)

        handle = layer.register_load_state_dict_pre_hook(hook)
        handle.remove()
        layer.load_state_dict(layer.state_dict())
        assert called == []


class TestErrorAccumulation:
    def test_multiple_size_mismatches_reported_together(self):
        # Build a model with two layers; produce a checkpoint where both
        # weight tensors have wrong shapes.
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
        sd = OrderedDict()
        sd["0.weight"] = lucid.zeros(99, 99)
        sd["0.bias"] = lucid.zeros(8)
        sd["1.weight"] = lucid.zeros(77, 77)
        sd["1.bias"] = lucid.zeros(4)
        sd._metadata = {}

        with pytest.raises(RuntimeError) as excinfo:
            model.load_state_dict(sd)
        # Both errors should appear.
        msg = str(excinfo.value)
        assert "0.weight" in msg
        assert "1.weight" in msg

    def test_strict_false_returns_incompatible(self):
        model = nn.Linear(4, 8)
        sd = OrderedDict()
        sd["weight"] = lucid.zeros(8, 4)
        sd["bogus"] = lucid.zeros(1)
        sd._metadata = {}
        result = model.load_state_dict(sd, strict=False)
        assert "bias" in result.missing_keys
        assert "bogus" in result.unexpected_keys


class TestBatchNormMigration:
    def test_num_batches_tracked_present_in_state_dict(self):
        bn = nn.BatchNorm2d(8)
        sd = bn.state_dict()
        assert "num_batches_tracked" in sd
        assert sd["num_batches_tracked"].dtype == lucid.int64

    def test_old_checkpoint_loads_without_num_batches_tracked(self):
        bn = nn.BatchNorm2d(8)
        sd = bn.state_dict()
        # Strip the new buffer & version metadata to mimic a v1 checkpoint.
        sd_old = OrderedDict((k, v) for k, v in sd.items() if "num_batches" not in k)
        sd_old._metadata = {}  # no version → migration path
        bn2 = nn.BatchNorm2d(8)
        result = bn2.load_state_dict(sd_old)
        assert result.missing_keys == []
        assert int(bn2.num_batches_tracked.numpy()) == 0

    def test_v2_checkpoint_loads_normally(self):
        src = nn.BatchNorm1d(4)
        dst = nn.BatchNorm1d(4)
        result = dst.load_state_dict(src.state_dict())
        assert result.missing_keys == []
        assert result.unexpected_keys == []


class TestLazyLinearWithNewProtocol:
    def test_lazy_linear_materializes_via_new_hook(self):
        src = nn.Linear(4, 8)
        dst = nn.LazyLinear(8)
        dst.load_state_dict(src.state_dict())
        assert dst.in_features == 4
        assert dst.weight.shape == (8, 4)


class TestDiskRoundTrip:
    def test_save_load_preserves_metadata(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
        sd = model.state_dict()
        original_meta = sd._metadata
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.lucid")
            lucid.save(sd, path)
            loaded = lucid.load(path, weights_only=False)
        assert hasattr(loaded, "_metadata")
        assert loaded._metadata == original_meta

    def test_save_load_full_roundtrip(self):
        src = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
        original = {k: v.numpy().copy() for k, v in src.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.lucid")
            lucid.save(src.state_dict(), path)
            loaded_sd = lucid.load(path, weights_only=False)
            dst = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8))
            dst.load_state_dict(loaded_sd)
        for k, original_arr in original.items():
            np.testing.assert_allclose(
                dst.state_dict()[k].numpy(), original_arr
            )

    def test_load_v1_checkpoint_no_metadata(self):
        # Manually craft a v1-style container (no _state_dict_metadata).
        layer = nn.Linear(4, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.lucid")
            # state_dict (an OrderedDict with _metadata) — but save it
            # via an unconditional path so we can simulate v1.  Easiest:
            # round-trip a plain dict without _metadata.
            plain = dict(layer.state_dict())  # drops _metadata
            lucid.save(plain, path)
            loaded = lucid.load(path, weights_only=False)
        assert isinstance(loaded, dict)
        # Should still load into a fresh module.
        layer2 = nn.Linear(4, 8)
        layer2.load_state_dict(loaded)
