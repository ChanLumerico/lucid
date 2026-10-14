"""``embedding`` and ``embedding_bag`` isolate or refuse out-of-range indices instead of reading past the table.

Found 2026-07-26 while sweeping the model zoo on Metal: feeding a model an
index tensor it did not expect killed the process with SIGSEGV rather than
being caught.  Before the engine bounds contract (LCD-228 / LCD-283), the
engine gather did no bounds checking, so an out-of-range index read whatever
followed the table — **zeros for a small overrun**, which is the dangerous
case, since a wrong ``vocab_size`` or an unclamped token id then trained
silently on empty embeddings; and a segfault once the offset was large.

CPU refuses an out-of-range index with ``IndexError`` before the gather runs;
Metal isolates it inside the gather (Policy B: NaN row for a floating table,
with zero host syncs).
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine

DEVICES = ["cpu", "metal"]


def _ids(values, device):
    return lucid.tensor(np.array(values, dtype=np.int64), device=device)


def _check_out_of_range(device, fn, *, bad_pos=0):
    if device == "cpu":
        with pytest.raises(IndexError, match="out of range"):
            fn()
        return
    syncs = _C_engine.host_sync_count()
    out = fn()
    assert _C_engine.host_sync_count() == syncs
    rows = out.numpy().reshape(-1, out.shape[-1])
    assert np.isnan(rows[bad_pos]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_in_range_lookup_matches_the_table(device):
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4).to(device)
    table = emb.weight.numpy()
    out = emb(_ids([[0, 9, 3]], device)).numpy()
    assert np.abs(out[0][0] - table[0]).max() == 0.0
    assert np.abs(out[0][1] - table[9]).max() == 0.0
    assert np.abs(out[0][2] - table[3]).max() == 0.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [10, 11, 999, 10**6])
def test_index_at_or_above_num_embeddings_raises(device, bad):
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4).to(device)
    ids = _ids([bad], device)
    _check_out_of_range(device, lambda: emb(ids))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [-1, -100])
def test_negative_index_raises(device, bad):
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4).to(device)
    ids = _ids([bad], device)
    _check_out_of_range(device, lambda: emb(ids))


@pytest.mark.parametrize("device", DEVICES)
def test_boundary_index_is_accepted(device):
    """``num_embeddings - 1`` is valid; only ``num_embeddings`` is not."""
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4).to(device)
    assert emb(_ids([9], device)).shape == (1, 4)
    ids = _ids([10], device)
    _check_out_of_range(device, lambda: emb(ids))


@pytest.mark.parametrize("device", DEVICES)
def test_functional_embedding_checks_too(device):
    lucid.manual_seed(0)
    table = lucid.randn(6, 3).to(device)
    assert F.embedding(_ids([[0, 5]], device), table).shape == (1, 2, 3)
    ids = _ids([[0, 6]], device)
    _check_out_of_range(device, lambda: F.embedding(ids, table), bad_pos=1)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_index_tensor_is_allowed(device):
    """An empty batch has no index to validate — must not raise."""
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4).to(device)
    out = emb(lucid.tensor(np.zeros((0,), dtype=np.int64), device=device))
    assert out.shape == (0, 4)


@pytest.mark.parametrize("device", DEVICES)
def test_padding_idx_still_works(device):
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4, padding_idx=0).to(device)
    assert emb(_ids([[0, 1, 2]], device)).shape == (1, 3, 4)


@pytest.mark.parametrize("device", DEVICES)
def test_training_through_embedding_is_unaffected(device):
    """The guard must not disturb the gradient path."""
    lucid.manual_seed(0)
    emb = nn.Embedding(10, 4).to(device)
    out = emb(_ids([[1, 2, 1]], device))
    (out * lucid.ones_like(out)).sum().backward()
    assert emb.weight.grad is not None
    grad = emb.weight.grad.numpy()
    # Row 1 appears twice, row 2 once, everything else untouched.
    assert np.allclose(grad[1], 2.0)
    assert np.allclose(grad[2], 1.0)
    assert np.allclose(grad[0], 0.0)


# ── EmbeddingBag shares the contract ─────────────────────────────────────────
# The first fix covered F.embedding only; EmbeddingBag went through a separate
# engine entry point and kept the unchecked gather.  Both now call the same
# check_embedding_indices helper so they cannot disagree on what is valid.


@pytest.mark.parametrize("device", DEVICES)
def test_embedding_bag_in_range(device):
    lucid.manual_seed(0)
    bag = nn.EmbeddingBag(10, 4).to(device)
    assert bag(_ids([[0, 1, 2]], device)).shape == (1, 4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [10, 999, -1])
def test_embedding_bag_rejects_out_of_range(device, bad):
    lucid.manual_seed(0)
    bag = nn.EmbeddingBag(10, 4).to(device)
    ids = _ids([[0, 1, bad]], device)
    _check_out_of_range(device, lambda: bag(ids))


@pytest.mark.parametrize("device", DEVICES)
def test_functional_embedding_bag_rejects_out_of_range(device):
    lucid.manual_seed(0)
    table = lucid.randn(6, 3).to(device)
    assert F.embedding_bag(_ids([[0, 5]], device), table).shape == (1, 3)
    ids = _ids([[0, 6]], device)
    _check_out_of_range(device, lambda: F.embedding_bag(ids, table))
