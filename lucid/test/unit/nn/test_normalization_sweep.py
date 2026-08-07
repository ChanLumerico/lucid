"""Every normalisation layer, across the options that select a path.

``normalization.py`` sat at 46.9%, and the dark lines were the running
statistics — the ``track_running_stats`` bookkeeping, the eval-time use
of those buffers, and the affine-free variants.  A layer was built in
train mode and called once; the half of the file that exists for
inference never ran.

The train/eval distinction is asserted on behaviour rather than on a
flag: a batch norm in eval mode has to give the *same answer for the same
input regardless of what else is in the batch*, which is the entire point
of keeping running statistics.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


def _x(shape):
    return lucid.tensor(
        np.random.default_rng(len(shape)).standard_normal(shape).astype(np.float32)
    )


BATCH_NORMS = [
    (nn.BatchNorm1d, (4, 6, 8)),
    (nn.BatchNorm2d, (4, 6, 5, 5)),
    (nn.BatchNorm3d, (4, 6, 3, 3, 3)),
]


# ── batch norm: the running statistics ────────────────────────────────────────


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_running_stats_move_in_training_and_hold_in_eval(cls, shape):
    layer = cls(6)
    before = np.asarray(layer.running_mean.numpy()).copy()
    layer(_x(shape))
    after_train = np.asarray(layer.running_mean.numpy()).copy()
    assert not np.allclose(before, after_train)

    layer.eval()
    layer(_x(shape))
    assert np.allclose(after_train, np.asarray(layer.running_mean.numpy()))


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_eval_output_does_not_depend_on_the_rest_of_the_batch(cls, shape):
    """The point of running statistics: at inference one sample's answer
    is its own, not a function of who it was batched with."""
    layer = cls(6)
    for _ in range(3):
        layer(_x(shape))
    layer.eval()

    full = _x(shape)
    alone = full[:1]
    from_batch = np.asarray(layer(full).numpy())[0]
    from_alone = np.asarray(layer(alone).numpy())[0]
    assert np.allclose(from_batch, from_alone, atol=1e-5)


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_batches_tracked_counts_up(cls, shape):
    layer = cls(6)
    assert int(np.asarray(layer.num_batches_tracked.numpy())) == 0
    layer(_x(shape))
    layer(_x(shape))
    assert int(np.asarray(layer.num_batches_tracked.numpy())) == 2


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_without_running_stats_eval_normalises_the_batch(cls, shape):
    """``track_running_stats=False`` has no buffers to fall back on, so
    eval must use the batch itself rather than silently using zeros."""
    layer = cls(6, track_running_stats=False)
    assert getattr(layer, "running_mean", None) is None
    layer.eval()
    out = np.asarray(layer(_x(shape)).numpy())
    assert np.isfinite(out).all()
    assert np.abs(out).max() > 0.0


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_affine_free_has_no_parameters(cls, shape):
    layer = cls(6, affine=False)
    assert list(layer.parameters()) == []
    assert np.isfinite(np.asarray(layer(_x(shape)).numpy())).all()


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_training_output_is_standardised(cls, shape):
    """Per channel, mean 0 and variance 1 — the definition."""
    layer = cls(6, affine=False)
    out = np.asarray(layer(_x(shape)).numpy())
    axes = (0,) + tuple(range(2, out.ndim))
    assert np.allclose(out.mean(axis=axes), 0.0, atol=1e-4)
    assert np.allclose(out.var(axis=axes), 1.0, atol=1e-2)


@pytest.mark.parametrize("momentum", [0.1, 0.5, 1.0])
def test_momentum_controls_how_fast_the_average_moves(momentum):
    layer = nn.BatchNorm1d(6, momentum=momentum)
    x = _x((4, 6, 8))
    layer(x)
    moved = np.abs(np.asarray(layer.running_mean.numpy())).sum()
    slower = nn.BatchNorm1d(6, momentum=momentum / 10.0)
    slower(x)
    assert moved > np.abs(np.asarray(slower.running_mean.numpy())).sum()


# ── the batch-free normalisers ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build,shape",
    [
        (lambda: nn.LayerNorm(8), (4, 8)),
        (lambda: nn.LayerNorm([5, 8]), (4, 5, 8)),
        (lambda: nn.GroupNorm(2, 6), (4, 6, 5)),
        (lambda: nn.GroupNorm(6, 6), (4, 6, 5)),  # per-channel, i.e. instance
        (lambda: nn.GroupNorm(1, 6), (4, 6, 5)),  # one group, i.e. layer
        (lambda: nn.InstanceNorm1d(6), (4, 6, 8)),
        (lambda: nn.InstanceNorm2d(6), (4, 6, 5, 5)),
        (lambda: nn.RMSNorm(8), (4, 8)),
    ],
)
def test_these_do_not_depend_on_the_batch(build, shape):
    """None of them keep running statistics, so a sample's answer must be
    identical alone and in company — in train mode as well as eval."""
    layer = build()
    full = _x(shape)
    together = np.asarray(layer(full).numpy())[0]
    alone = np.asarray(layer(full[:1]).numpy())[0]
    assert np.allclose(together, alone, atol=1e-5)


def test_group_norm_with_one_group_matches_layer_norm_over_the_same_axes():
    """A single group is a layer norm; if they disagree, one is wrong."""
    x = _x((4, 6, 5))
    grouped = np.asarray(nn.GroupNorm(1, 6, affine=False)(x).numpy())
    flat = np.asarray(x.numpy()).reshape(4, -1)
    expected = (flat - flat.mean(1, keepdims=True)) / np.sqrt(
        flat.var(1, keepdims=True) + 1e-5
    )
    assert np.allclose(grouped.reshape(4, -1), expected, atol=1e-4)


def test_rms_norm_does_not_subtract_the_mean():
    """That is the whole difference from a layer norm: scale only."""
    x = lucid.tensor(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    layer = nn.RMSNorm(4)
    out = np.asarray(layer(x).numpy())
    assert out.mean() > 0.1  # a layer norm would centre this at zero
    # ``RMSNorm`` here takes only ``normalized_shape`` and ``eps`` — the
    # reference also offers ``elementwise_affine``, which this does not.
    rms = np.sqrt((np.asarray(x.numpy()) ** 2).mean())
    weight = np.asarray(layer.weight.numpy()) if hasattr(layer, "weight") else 1.0
    assert np.allclose(out, np.asarray(x.numpy()) / rms * weight, atol=1e-3)


@pytest.mark.parametrize("elementwise_affine", [True, False])
def test_layer_norm_affine_flag(elementwise_affine):
    layer = nn.LayerNorm(8, elementwise_affine=elementwise_affine)
    assert (len(list(layer.parameters())) > 0) == elementwise_affine
    assert np.isfinite(np.asarray(layer(_x((4, 8))).numpy())).all()


# ── state and gradients ───────────────────────────────────────────────────────


@pytest.mark.parametrize("cls,shape", BATCH_NORMS)
def test_running_stats_survive_a_state_dict_round_trip(cls, shape):
    source = cls(6)
    for _ in range(3):
        source(_x(shape))
    target = cls(6)
    target.load_state_dict(source.state_dict())
    assert np.allclose(
        np.asarray(source.running_mean.numpy()),
        np.asarray(target.running_mean.numpy()),
    )
    source.eval()
    target.eval()
    x = _x(shape)
    assert np.allclose(
        np.asarray(source(x).numpy()), np.asarray(target(x).numpy()), atol=1e-6
    )


@pytest.mark.parametrize(
    "build,shape",
    [
        (lambda: nn.BatchNorm1d(6), (4, 6, 8)),
        (lambda: nn.LayerNorm(8), (4, 8)),
        (lambda: nn.GroupNorm(2, 6), (4, 6, 5)),
        (lambda: nn.RMSNorm(8), (4, 8)),
    ],
)
def test_gradients_reach_the_affine_parameters(build, shape):
    layer = build()
    layer(_x(shape)).sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name


def test_repr_names_the_shape():
    assert "8" in repr(nn.LayerNorm(8))
    assert "6" in repr(nn.BatchNorm1d(6))
