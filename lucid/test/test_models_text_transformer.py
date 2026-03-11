import pytest

import lucid

from lucid.models import (
    Transformer,
    TransformerConfig,
    transformer_base,
    transformer_big,
)


def test_transformer_public_imports() -> None:
    assert Transformer is not None
    assert TransformerConfig is not None
    assert transformer_base is not None
    assert transformer_big is not None


def test_transformer_accepts_config_and_runs_forward() -> None:
    config = TransformerConfig(
        src_vocab_size=32,
        tgt_vocab_size=40,
        d_model=8,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_len=16,
    )

    model = Transformer(config)
    src = lucid.Tensor([[1, 2], [3, 4], [5, 6]], dtype=lucid.Int32)
    tgt = lucid.Tensor([[1, 2], [3, 4], [5, 6]], dtype=lucid.Int32)

    out = model(src, tgt)

    assert model.config is config
    assert out.shape == (3, 2, 40)


def test_transformer_factories_build_variants() -> None:
    model_base = transformer_base(
        src_vocab_size=32,
        tgt_vocab_size=40,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_len=16,
    )
    model_big = transformer_big(
        src_vocab_size=32,
        tgt_vocab_size=40,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_len=16,
    )

    src = lucid.Tensor([[1], [2], [3]], dtype=lucid.Int32)
    tgt = lucid.Tensor([[1], [2], [3]], dtype=lucid.Int32)

    out_base = model_base(src, tgt)
    out_big = model_big(src, tgt)

    assert model_base.config.d_model == 512
    assert model_big.config.d_model == 1024
    assert out_base.shape == (3, 1, 40)
    assert out_big.shape == (3, 1, 40)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"src_vocab_size": 0},
        {"tgt_vocab_size": 0},
        {"d_model": 0},
        {"num_heads": 0},
        {"d_model": 7, "num_heads": 2},
        {"num_encoder_layers": 0},
        {"num_decoder_layers": 0},
        {"dim_feedforward": 0},
        {"dropout": 1.0},
        {"max_len": 0},
    ),
)
def test_transformer_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    params = {
        "src_vocab_size": 32,
        "tgt_vocab_size": 40,
        "d_model": 8,
        "num_heads": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 16,
    }
    params.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        TransformerConfig(**params)
