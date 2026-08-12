"""Transformer paths the suite never entered.

The one that matters is checkpoint compatibility.  The stacks' layers
moved under a ``layers.`` submodule, which changed every state-dict key
from ``encoder.0.self_attn.…`` to ``encoder.layers.0.self_attn.…``, and
``_lift_flat_layer_keys`` exists so files written before that move still
load.  Nothing exercised it, so the guarantee was resting on code no test
had ever run — and a compatibility shim that is never tested is a
compatibility claim, not a compatibility feature.

The rest is the small surface around it: the container that holds the
layers, the prototype accessors used to place lazily-built layers, and
the ``extra_repr`` of each stack, which is what a user sees when they
print a model.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


def _encoder(num_layers: int = 2) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def _decoder(num_layers: int = 2) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    return nn.TransformerDecoder(layer, num_layers=num_layers)


# ── the flat-checkpoint shim ─────────────────────────────────────────────────


def _flatten_keys(state: dict, prefix: str) -> dict:
    """Rewrite ``<prefix>layers.N.*`` back to the pre-move ``<prefix>N.*``."""
    out = {}
    for key, value in state.items():
        marker = f"{prefix}layers."
        if key.startswith(marker):
            out[f"{prefix}{key[len(marker):]}"] = value
        else:
            out[key] = value
    return out


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_a_pre_layers_checkpoint_still_loads(kind: str) -> None:
    """The old flat key layout must load and produce the same output."""
    model = _encoder() if kind == "encoder" else _decoder()
    lucid.manual_seed(0)
    src = lucid.randn(2, 5, 8)
    args = (src,) if kind == "encoder" else (src, lucid.randn(2, 5, 8))

    model.eval()
    want = model(*args).numpy()

    flat = _flatten_keys(dict(model.state_dict()), "")
    assert any(k.startswith("layers.") for k in model.state_dict())
    assert not any(k.startswith("layers.") for k in flat), "the fixture did not flatten"

    fresh = _encoder() if kind == "encoder" else _decoder()
    fresh.load_state_dict(flat)
    fresh.eval()
    got = fresh(*args).numpy()

    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_the_shim_leaves_a_current_checkpoint_alone() -> None:
    """A file already in the new layout must round-trip untouched."""
    model = _encoder()
    lucid.manual_seed(0)
    src = lucid.randn(2, 5, 8)
    model.eval()
    want = model(src).numpy()

    fresh = _encoder()
    fresh.load_state_dict(dict(model.state_dict()))
    fresh.eval()
    np.testing.assert_allclose(fresh(src).numpy(), want, rtol=1e-5, atol=1e-5)


def test_the_shim_does_not_clobber_an_existing_key() -> None:
    """When both layouts are present the current one wins.

    ``_lift_flat_layer_keys`` only writes a lifted key that is not
    already there; a file carrying both spellings must not have its real
    entry replaced by the legacy one.
    """
    from lucid.nn.modules.transformer import _lift_flat_layer_keys

    current = lucid.ones(2)
    legacy = lucid.zeros(2)
    state = {"layers.0.w": current, "0.w": legacy}
    _lift_flat_layer_keys(state, "")

    assert state["layers.0.w"] is current, "the legacy key overwrote the current one"


# ── the layer container ──────────────────────────────────────────────────────


def test_the_container_behaves_like_the_list_it_replaced() -> None:
    model = _encoder(num_layers=3)
    layers = model.layers
    assert len(layers) == 3
    assert layers[0] is list(layers)[0]
    assert layers[2] is list(layers)[2]
    assert [id(x) for x in layers] == [id(layers[i]) for i in range(3)]


def test_the_container_puts_layers_into_the_state_dict_keys() -> None:
    """The whole reason the container exists."""
    keys = list(_encoder(num_layers=2).state_dict())
    assert any(k.startswith("layers.0.") for k in keys), keys[:4]
    assert any(k.startswith("layers.1.") for k in keys), keys[:4]


# ── prototype accessors ──────────────────────────────────────────────────────


def test_the_prototype_device_and_dtype_come_off_the_first_parameter() -> None:
    from lucid.nn.modules.transformer import _proto_device, _proto_dtype

    layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    first = next(iter(layer.parameters()))
    assert _proto_device(layer) == first.device.type
    assert _proto_dtype(layer) == first.dtype


def test_a_parameterless_prototype_reports_nothing() -> None:
    """No parameters means no device or dtype to copy — not a crash."""
    from lucid.nn.modules.transformer import _proto_device, _proto_dtype

    empty = nn.Identity()
    assert _proto_device(empty) is None
    assert _proto_dtype(empty) is None


# ── extra_repr ───────────────────────────────────────────────────────────────


def test_every_stack_describes_itself() -> None:
    """``print(model)`` has to say something true about each piece."""
    enc_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    dec_layer = nn.TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16)

    for layer in (enc_layer, dec_layer):
        text = layer.extra_repr()
        assert "d_model=8" in text and "nhead=2" in text, text
        assert "dim_feedforward=16" in text, text

    for stack in (_encoder(num_layers=3), _decoder(num_layers=3)):
        assert stack.extra_repr() == "num_layers=3"

    whole = nn.Transformer(d_model=8, nhead=2, dim_feedforward=16)
    assert whole.extra_repr() == "d_model=8, nhead=2"
