"""
Tests for lucid.einops, lucid.serialization, and lucid.profiler.
"""

import io
import os
import tempfile
import numpy as np
import pytest

import lucid
import lucid.einops as einops
import lucid.profiler as profiler
import lucid.serialization as serialization
import lucid.nn as nn


# ── lucid.einops ──────────────────────────────────────────────────────────────


class TestRearrange:
    def test_transpose_2d(self):
        x = lucid.randn(3, 4)
        y = einops.rearrange(x, "a b -> b a")
        assert y.shape == (4, 3)
        np.testing.assert_allclose(y.numpy(), x.numpy().T, atol=1e-6)

    def test_flatten(self):
        x = lucid.randn(2, 3, 4)
        y = einops.rearrange(x, "a b c -> a (b c)")
        assert y.shape == (2, 12)

    def test_unflatten(self):
        x = lucid.randn(2, 12)
        y = einops.rearrange(x, "a (b c) -> a b c", b=3)
        assert y.shape == (2, 3, 4)

    def test_add_batch_dim(self):
        x = lucid.randn(3, 4)
        y = einops.rearrange(x, "h w -> 1 h w")
        assert y.shape == (1, 3, 4)

    def test_permute_3d(self):
        x = lucid.randn(2, 3, 4)
        y = einops.rearrange(x, "b h w -> b w h")
        assert y.shape == (2, 4, 3)

    def test_merge_and_split(self):
        x = lucid.randn(2, 6)
        y = einops.rearrange(x, "b (h w) -> b h w", h=2)
        assert y.shape == (2, 2, 3)
        z = einops.rearrange(y, "b h w -> b (h w)")
        assert z.shape == (2, 6)
        np.testing.assert_allclose(z.numpy(), x.numpy(), atol=1e-6)


class TestReduce:
    def test_mean_spatial(self):
        x = lucid.randn(2, 3, 4)
        y = einops.reduce(x, "b h w -> b", "mean")
        assert y.shape == (2,)

    def test_sum_channel(self):
        x = lucid.ones(2, 4, 4)
        y = einops.reduce(x, "b h w -> b h", "sum")
        assert y.shape == (2, 4)
        np.testing.assert_allclose(y.numpy(), np.full((2, 4), 4.0), atol=1e-5)

    def test_max_reduction(self):
        x = lucid.tensor([[[1.0, 3.0], [2.0, 4.0]]])  # (1, 2, 2)
        y = einops.reduce(x, "b h w -> b", "max")
        assert abs(float(y.item()) - 4.0) < 1e-5

    def test_global_avg_pool(self):
        x = lucid.ones(4, 8, 8)  # (B, H, W)
        y = einops.reduce(x, "b h w -> b", "mean")
        np.testing.assert_allclose(y.numpy(), np.ones(4), atol=1e-5)


class TestRepeat:
    def test_repeat_new_axis(self):
        x = lucid.randn(3, 4)
        y = einops.repeat(x, "h w -> b h w", b=5)
        assert y.shape == (5, 3, 4)

    def test_repeat_tile(self):
        x = lucid.tensor([[1.0, 2.0]])
        y = einops.repeat(x, "1 w -> n w", n=3)
        assert y.shape == (3, 2)
        expected = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        np.testing.assert_allclose(y.numpy(), expected, atol=1e-6)


class TestEinsum:
    def test_matmul(self):
        A = lucid.randn(3, 4)
        B = lucid.randn(4, 5)
        C = einops.einsum("ij,jk->ik", A, B)
        expected = lucid.matmul(A, B)
        np.testing.assert_allclose(C.numpy(), expected.numpy(), atol=1e-5)

    def test_dot_product(self):
        a = lucid.tensor([1.0, 2.0, 3.0])
        b = lucid.tensor([4.0, 5.0, 6.0])
        result = einops.einsum("i,i->", a, b)
        assert abs(float(result.item()) - 32.0) < 1e-4

    def test_batch_matmul(self):
        A = lucid.randn(2, 3, 4)
        B = lucid.randn(2, 4, 5)
        C = einops.einsum("bij,bjk->bik", A, B)
        assert C.shape == (2, 3, 5)

    def test_outer_product(self):
        a = lucid.tensor([1.0, 2.0])
        b = lucid.tensor([3.0, 4.0, 5.0])
        C = einops.einsum("i,j->ij", a, b)
        assert C.shape == (2, 3)
        expected = np.outer([1.0, 2.0], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(C.numpy(), expected, atol=1e-5)


# ── lucid.serialization ───────────────────────────────────────────────────────


class TestSaveTensor:
    def test_save_load_tensor_filepath(self):
        t = lucid.randn(3, 4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            serialization.save(t, path)
            t2 = serialization.load(path)
            np.testing.assert_allclose(t2.numpy(), t.numpy(), atol=1e-6)
        finally:
            os.unlink(path)

    def test_save_load_tensor_fileobj(self):
        t = lucid.tensor([1.0, 2.0, 3.0])
        buf = io.BytesIO()
        serialization.save(t, buf)
        buf.seek(0)
        t2 = serialization.load(buf)
        np.testing.assert_allclose(t2.numpy(), t.numpy(), atol=1e-6)

    def test_save_load_dict_of_tensors(self):
        data = {"a": lucid.randn(2, 3), "b": lucid.randn(4)}
        buf = io.BytesIO()
        serialization.save(data, buf)
        buf.seek(0)
        data2 = serialization.load(buf)
        assert set(data2.keys()) == {"a", "b"}
        np.testing.assert_allclose(data2["a"].numpy(), data["a"].numpy(), atol=1e-6)

    def test_save_load_python_scalars(self):
        obj = {"lr": 1e-3, "epoch": 42, "name": "model"}
        buf = io.BytesIO()
        serialization.save(obj, buf)
        buf.seek(0)
        obj2 = serialization.load(buf)
        assert obj2 == obj

    def test_save_load_list_of_tensors(self):
        lst = [lucid.randn(2), lucid.randn(3)]
        buf = io.BytesIO()
        serialization.save(lst, buf)
        buf.seek(0)
        lst2 = serialization.load(buf)
        assert len(lst2) == 2
        np.testing.assert_allclose(lst2[0].numpy(), lst[0].numpy(), atol=1e-6)


class TestSaveLoadModel:
    def test_state_dict_round_trip(self):
        m = nn.Linear(4, 2)
        sd = m.state_dict()
        buf = io.BytesIO()
        serialization.save(sd, buf)
        buf.seek(0)
        sd2 = serialization.load(buf)
        np.testing.assert_allclose(
            sd2["weight"].numpy(), sd["weight"].numpy(), atol=1e-6
        )

    def test_full_model_round_trip(self):
        m = nn.Linear(3, 2)
        x = lucid.randn(4, 3)
        out_before = m(x).numpy().copy()
        buf = io.BytesIO()
        serialization.save(m.state_dict(), buf)
        buf.seek(0)
        sd = serialization.load(buf)
        m2 = nn.Linear(3, 2)
        m2.load_state_dict(sd)
        out_after = m2(x).numpy()
        np.testing.assert_allclose(out_before, out_after, atol=1e-5)

    def test_save_to_bytes_path(self):
        t = lucid.tensor([42.0])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name.encode()  # bytes path
        try:
            serialization.save(t, path)
            t2 = serialization.load(path)
            assert abs(float(t2.item()) - 42.0) < 1e-5
        finally:
            os.unlink(path.decode())


# ── lucid.profiler ────────────────────────────────────────────────────────────


class TestProfiler:
    def test_context_manager_records_events(self):
        with profiler.Profiler() as prof:
            x = lucid.randn(10)
            _ = lucid.sum(x)
        events = prof.events()
        assert len(events) >= 2  # randn + sum

    def test_op_event_fields(self):
        with profiler.Profiler() as prof:
            _ = lucid.relu(lucid.randn(5))
        events = prof.events()
        assert len(events) >= 1
        ev = events[-1]
        assert isinstance(ev.name, str)
        assert ev.time_us >= 0
        assert ev.time_ms >= 0
        assert isinstance(ev.shape, list)

    def test_key_averages(self):
        with profiler.Profiler() as prof:
            _ = lucid.randn(4)
            _ = lucid.randn(4)
        summaries = prof.key_averages()
        assert len(summaries) >= 1
        s = summaries[0]
        assert isinstance(s.name, str)
        assert s.count >= 1

    def test_profiler_no_memory(self):
        with profiler.Profiler(with_memory=False) as prof:
            _ = lucid.randn(3) * 2.0
        # with_memory=False: still records events

    def test_record_function(self):
        with profiler.Profiler() as prof:
            with profiler.record_function("my_custom_op"):
                _ = lucid.sum(lucid.randn(4))
        # Should complete without raising

    def test_profile_context_manager(self):
        with profiler.profile() as prof:
            _ = lucid.mul(lucid.randn(3), lucid.randn(3))
        assert isinstance(prof, profiler.Profiler)
        assert len(prof.events()) >= 1

    def test_memory_stats(self):
        with profiler.Profiler() as prof:
            _ = lucid.randn(100)
        stats = prof.memory_stats()
        # Returns None or a MemoryStats object — both valid
        assert stats is None or hasattr(stats, "current_bytes") or hasattr(stats, "current")

    def test_clear(self):
        with profiler.Profiler() as prof:
            _ = lucid.randn(3)
        assert len(prof.events()) >= 1
        prof.clear()
        assert len(prof.events()) == 0
