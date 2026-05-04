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
        model = nn.Transformer(d_model=16, nhead=4, num_encoder_layers=2,
                               num_decoder_layers=2, batch_first=True)
        src = make_tensor((4, 8, 16))
        tgt = make_tensor((4, 6, 16))
        out = model(src, tgt)
        assert out.shape == (4, 6, 16)
