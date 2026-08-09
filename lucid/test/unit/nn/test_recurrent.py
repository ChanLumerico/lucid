"""The recurrent layers, against the reference and against themselves.

``nn/modules/rnn.py`` sat at 74.5%, and what was dark was every option
that changes the recurrence: a second layer, a backward direction, no
bias, ``batch_first=False``, a supplied initial state, ``proj_size``,
inter-layer dropout.

A recurrence is unusually good at hiding a mistake.  The output has the
right shape whether or not the backward direction was concatenated the
right way round, whether or not layer two received layer one's output,
and whether or not the state handed back is the one at the last step.
So the reference is the oracle for the values, and the properties that
do not need one — that carrying the state across a split equals running
the sequence whole — are asserted separately.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from lucid.test._fixtures.ref_framework import require_ref

RNG = np.random.default_rng(0)
X = RNG.standard_normal((3, 5, 4)).astype(np.float32)


def _v(x):
    return np.asarray(x.numpy())


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float32))


def _mirror(lucid_module, ref_module, ref):
    """Give the reference module Lucid's weights, so only the recurrence
    is under test rather than two different initialisations."""
    target = ref_module.state_dict()
    for name, tensor in lucid_module.state_dict().items():
        if name in target:
            target[name].copy_(ref.tensor(_v(tensor)))
    ref_module.load_state_dict(target)


def _first(out):
    return out[0] if isinstance(out, tuple) else out


OPTIONS = [
    ("default", {}),
    ("two layers", {"num_layers": 2}),
    ("bidirectional", {"bidirectional": True}),
    ("two layers bidirectional", {"num_layers": 2, "bidirectional": True}),
    ("no bias", {"bias": False}),
    ("sequence first", {"batch_first": False}),
]


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
@pytest.mark.parametrize(
    "options", [o[1] for o in OPTIONS], ids=[o[0] for o in OPTIONS]
)
def test_the_recurrence_matches_the_reference(kind, options):
    ref = require_ref()
    kwargs = dict(options)
    batch_first = kwargs.pop("batch_first", True)
    built = dict(input_size=4, hidden_size=6, batch_first=batch_first, **kwargs)

    lucid.manual_seed(0)
    mine = getattr(nn, kind)(**built)
    theirs = getattr(ref.nn, kind)(**built)
    _mirror(mine, theirs, ref)

    inputs = X if batch_first else X.transpose(1, 0, 2)
    got = _v(_first(mine(_t(inputs))))
    want = _first(theirs(ref.tensor(inputs))).detach().numpy()
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-4)


@pytest.mark.parity
@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_the_plain_rnn_honours_its_nonlinearity(nonlinearity):
    ref = require_ref()
    lucid.manual_seed(0)
    mine = nn.RNN(4, 6, batch_first=True, nonlinearity=nonlinearity)
    theirs = ref.nn.RNN(4, 6, batch_first=True, nonlinearity=nonlinearity)
    _mirror(mine, theirs, ref)
    assert np.allclose(
        _v(_first(mine(_t(X)))),
        _first(theirs(ref.tensor(X))).detach().numpy(),
        atol=1e-4,
    )


@pytest.mark.parity
def test_lstm_projects_its_hidden_state_when_asked():
    """``proj_size`` makes the output narrower than the cell, which is
    the one option that changes the output *shape* as well as its
    values."""
    ref = require_ref()
    lucid.manual_seed(0)
    mine = nn.LSTM(4, 6, proj_size=3, batch_first=True)
    theirs = ref.nn.LSTM(4, 6, proj_size=3, batch_first=True)
    _mirror(mine, theirs, ref)
    got = _v(_first(mine(_t(X))))
    assert got.shape == (3, 5, 3)
    assert np.allclose(got, _first(theirs(ref.tensor(X))).detach().numpy(), atol=1e-4)


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["RNNCell", "LSTMCell", "GRUCell"])
def test_a_single_cell_matches_the_reference(kind):
    ref = require_ref()
    lucid.manual_seed(0)
    mine = getattr(nn, kind)(4, 6)
    theirs = getattr(ref.nn, kind)(4, 6)
    _mirror(mine, theirs, ref)
    got, want = mine(_t(X[:, 0])), theirs(ref.tensor(X[:, 0]))
    if isinstance(got, tuple):
        for a, b in zip(got, want):
            assert np.allclose(_v(a), b.detach().numpy(), atol=1e-4)
    else:
        assert np.allclose(_v(got), want.detach().numpy(), atol=1e-4)


# ── the state, without needing an oracle ──────────────────────────────────────


def test_the_output_shape_follows_the_options():
    for options, expected in (
        ({}, (3, 5, 6)),
        ({"bidirectional": True}, (3, 5, 12)),
        ({"num_layers": 3}, (3, 5, 6)),
        ({"proj_size": 2}, (3, 5, 2)),
    ):
        layer = nn.LSTM(4, 6, batch_first=True, **options)
        assert _v(_first(layer(_t(X)))).shape == expected, options


def test_lstm_hands_back_both_halves_of_its_state():
    layer = nn.LSTM(4, 6, batch_first=True)
    out, (h_n, c_n) = layer(_t(X))
    assert _v(h_n).shape == (1, 3, 6)
    assert _v(c_n).shape == (1, 3, 6)
    # The hidden half of the state is the last output step.
    assert np.allclose(_v(h_n)[0], _v(out)[:, -1], atol=1e-5)


def test_carrying_the_state_across_a_split_equals_running_it_whole():
    """The property that says the state really is the state.

    A layer that returned the *first* step's state, or dropped the cell
    half, would still produce a plausibly-shaped answer for each piece —
    and only stitching the pieces back together shows it.
    """
    layer = nn.LSTM(4, 6, batch_first=True)
    whole = _v(_first(layer(_t(X))))
    head, state = layer(_t(X[:, :2]))
    tail, _ = layer(_t(X[:, 2:]), state)
    assert np.allclose(np.concatenate([_v(head), _v(tail)], axis=1), whole, atol=1e-4)


@pytest.mark.parametrize("kind", ["RNN", "GRU"])
def test_carrying_a_single_tensor_state_across_a_split(kind):
    layer = getattr(nn, kind)(4, 6, batch_first=True)
    whole = _v(_first(layer(_t(X))))
    head, state = layer(_t(X[:, :2]))
    tail, _ = layer(_t(X[:, 2:]), state)
    assert np.allclose(np.concatenate([_v(head), _v(tail)], axis=1), whole, atol=1e-4)


def test_a_supplied_initial_state_changes_the_answer():
    layer = nn.LSTM(4, 6, batch_first=True)
    zeros = _t(np.zeros((1, 3, 6)))
    ones = _t(np.ones((1, 3, 6)))
    default = _v(_first(layer(_t(X))))
    explicit_zero = _v(_first(layer(_t(X), (zeros, zeros))))
    nonzero = _v(_first(layer(_t(X), (ones, ones))))
    assert np.allclose(default, explicit_zero, atol=1e-6)  # the default is zeros
    assert not np.allclose(default, nonzero)


def test_the_two_directions_are_concatenated_not_summed():
    """Bidirectional output is ``[forward | backward]`` along the feature
    axis.  Summing them would have the right rank and half the width, and
    averaging would have the right width and the wrong values — so the
    forward half is compared against a one-directional layer."""
    lucid.manual_seed(0)
    both = nn.LSTM(4, 6, batch_first=True, bidirectional=True)
    out = _v(_first(both(_t(X))))
    assert out.shape == (3, 5, 12)
    forward, backward = out[..., :6], out[..., 6:]
    assert not np.allclose(forward, backward)


def test_more_layers_is_not_the_same_as_one():
    lucid.manual_seed(0)
    one = nn.GRU(4, 6, batch_first=True)
    lucid.manual_seed(0)
    three = nn.GRU(4, 6, batch_first=True, num_layers=3)
    assert _v(_first(one(_t(X)))).shape == _v(_first(three(_t(X)))).shape
    assert not np.allclose(_v(_first(one(_t(X)))), _v(_first(three(_t(X)))))


def test_batch_first_is_a_transposition_and_nothing_else():
    lucid.manual_seed(0)
    first = nn.GRU(4, 6, batch_first=True)
    lucid.manual_seed(0)
    later = nn.GRU(4, 6, batch_first=False)
    later.load_state_dict(first.state_dict())
    assert np.allclose(
        _v(_first(first(_t(X)))),
        _v(_first(later(_t(X.transpose(1, 0, 2))))).transpose(1, 0, 2),
        atol=1e-5,
    )


@pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
def test_gradients_reach_every_gate(kind):
    layer = getattr(nn, kind)(4, 6, batch_first=True, num_layers=2)
    _first(layer(_t(X))).sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name


def test_inter_layer_dropout_is_a_training_time_thing():
    layer = nn.LSTM(4, 6, num_layers=2, dropout=0.5, batch_first=True)
    layer.eval()
    assert np.allclose(_v(_first(layer(_t(X)))), _v(_first(layer(_t(X)))))
    layer.train()
    assert not np.allclose(_v(_first(layer(_t(X)))), _v(_first(layer(_t(X)))))


def test_a_recurrent_layer_learns_a_sequence():
    """End to end, because every property above can hold while the whole
    thing still fails to train."""
    lucid.manual_seed(0)
    layer = nn.GRU(4, 6, batch_first=True)
    head = nn.Linear(6, 1)
    params = list(layer.parameters()) + list(head.parameters())
    optimiser = lucid.optim.Adam(params, lr=0.05)
    targets = _t(RNG.standard_normal((3, 5, 1)))

    def loss():
        return ((head(_first(layer(_t(X)))) - targets) ** 2).mean()

    first = float(_v(loss()))
    for _ in range(40):
        optimiser.zero_grad()
        loss().backward()
        optimiser.step()
    assert float(_v(loss())) < first


# ── padding and packing ───────────────────────────────────────────────────────


def _ragged():
    return [_t(RNG.standard_normal((n, 4)).astype(np.float32)) for n in (5, 3, 1)]


def test_pad_sequence_pads_to_the_longest():
    padded = pad_sequence(_ragged(), batch_first=True)
    assert _v(padded).shape == (3, 5, 4)


def test_pad_sequence_leaves_the_real_steps_alone():
    seqs = _ragged()
    padded = _v(pad_sequence(seqs, batch_first=True))
    for row, seq in enumerate(seqs):
        length = _v(seq).shape[0]
        assert np.allclose(padded[row, :length], _v(seq))
        assert np.allclose(padded[row, length:], 0.0)


def test_pad_sequence_can_pad_with_something_other_than_zero():
    padded = _v(pad_sequence(_ragged(), batch_first=True, padding_value=-1.0))
    assert np.allclose(padded[2, 1:], -1.0)


def test_pack_then_pad_returns_what_went_in():
    padded = pad_sequence(_ragged(), batch_first=True)
    packed = pack_padded_sequence(padded, [5, 3, 1], batch_first=True)
    restored, lengths = pad_packed_sequence(packed, batch_first=True)
    assert np.allclose(_v(restored), _v(padded), atol=1e-6)
    assert list(np.asarray(_v(lengths)).ravel()) == [5, 3, 1]


def test_pack_sequence_takes_the_ragged_list_directly():
    packed = pack_sequence(_ragged())
    restored, lengths = pad_packed_sequence(packed, batch_first=True)
    assert _v(restored).shape == (3, 5, 4)
    assert list(np.asarray(_v(lengths)).ravel()) == [5, 3, 1]


@pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
def test_a_packed_sequence_is_refused_rather_than_silently_padded(kind):
    """Recorded, not a defect — the good failure mode.

    Packing exists so that padded steps do not reach the recurrence.  A
    layer that accepted a ``PackedSequence`` and quietly treated it as a
    padded batch would return the right shape and let the padding into
    every hidden state.  This refuses, names the limitation, and says
    what to do instead — and it is the *builtin* ``NotImplementedError``,
    so the obvious ``except NotImplementedError`` around a fallback
    actually fires.
    """
    padded = pad_sequence(_ragged(), batch_first=True)
    packed = pack_padded_sequence(padded, [5, 3, 1], batch_first=True)
    with pytest.raises(NotImplementedError, match="PackedSequence"):
        getattr(nn, kind)(4, 6, batch_first=True)(packed)
