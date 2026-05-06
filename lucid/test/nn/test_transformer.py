"""Tests for Transformer encoder/decoder."""

import pytest
import lucid
import lucid.nn as nn
from lucid.test.helpers.numerics import make_tensor


class TestTransformerEncoder:
    def test_output_shape(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
        encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        src = make_tensor((4, 8, 16))
        out = encoder(src)
        assert out.shape == (4, 8, 16)

    def test_single_layer(self):
        enc_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
        x = make_tensor((2, 5, 16))
        out = enc_layer(x)
        assert out.shape == (2, 5, 16)


class TestTransformerDecoder:
    def test_output_shape(self):
        dec_layer = nn.TransformerDecoderLayer(d_model=16, nhead=4, batch_first=True)
        decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        tgt = make_tensor((4, 6, 16))
        memory = make_tensor((4, 8, 16))
        out = decoder(tgt, memory)
        assert out.shape == (4, 6, 16)


class TestTransformer:
    def test_full_forward(self):
        model = nn.Transformer(
            d_model=16,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True,
        )
        src = make_tensor((4, 8, 16))
        tgt = make_tensor((4, 6, 16))
        out = model(src, tgt)
        assert out.shape == (4, 6, 16)


class TestTransformerKeyPaddingMask:
    """The key_padding_mask now flows through the encoder/decoder stack."""

    def test_encoder_layer_kpm_zeroes_attention_to_padded_keys(self):
        # Build a layer whose self-attention output is sensitive to keys.
        layer = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, batch_first=True
        )
        x = make_tensor((1, 4, 8))
        # Mask the last key — output for any query should not depend on it.
        kpm = lucid.tensor([[False, False, False, True]], dtype=lucid.bool_)
        # Two forwards: one with the padded key zeroed in input, one with garbage there.
        x_zeroed = x.numpy().copy()
        x_zeroed[:, 3, :] = 0.0
        x_garbage = x.numpy().copy()
        x_garbage[:, 3, :] = 1e6  # huge values that would dominate without mask.
        layer.eval()
        y_zeroed = layer(lucid.tensor(x_zeroed), src_key_padding_mask=kpm)
        y_garbage = layer(lucid.tensor(x_garbage), src_key_padding_mask=kpm)
        # Outputs at the unmasked query positions should match between the two
        # — the mask is doing its job by ignoring index 3.
        import numpy as np

        np.testing.assert_allclose(
            y_zeroed.numpy()[:, :3, :], y_garbage.numpy()[:, :3, :], atol=1e-4
        )
