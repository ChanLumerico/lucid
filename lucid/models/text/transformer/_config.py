"""Original Transformer (encoder-decoder) configuration — Vaswani et al., 2017.

Paper: "Attention Is All You Need".  This is the *seq2seq* transformer that
spawned the entire family — encoder + decoder + sinusoidal positional
encoding + shared (or split) source / target vocabularies.

The config inherits from :class:`LanguageModelConfig` (vocab_size = source
side) and adds the seq2seq-specific knobs:

    * ``decoder_vocab_size``    — target vocabulary; ``None`` means shared.
    * ``num_decoder_layers``    — paper uses 6 (same as encoder).
    * ``share_embeddings``      — tie source / target token embeddings.
    * ``tie_word_embeddings``   — tie LM head to the *target* embedding.
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models.text._config import LanguageModelConfig


@dataclass(frozen=True)
class TransformerConfig(LanguageModelConfig):
    """Configuration for every Vaswani-style Transformer variant.

    The bare paper specifies a single 65 537-token shared BPE vocabulary for
    WMT En-De and separate vocabularies for some tasks; we make this
    configurable via ``decoder_vocab_size`` / ``share_embeddings`` so callers
    can match whichever checkpoint they're porting.
    """

    model_type: ClassVar[str] = "transformer"

    # Vaswani et al. (2017) "base" defaults — Table 3.
    vocab_size: int = 37_000  # source side; paper's En-De WMT BPE
    hidden_size: int = 512  # d_model
    num_hidden_layers: int = 6  # encoder layers (paper N=6)
    num_attention_heads: int = 8  # h=8
    intermediate_size: int = 2_048  # d_ff
    max_position_embeddings: int = 5_000  # sinusoidal table size
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1

    # Seq2seq additions.
    decoder_vocab_size: int | None = None  # None → reuse vocab_size
    num_decoder_layers: int = 6  # paper N=6 on both sides
    share_embeddings: bool = False  # share src / tgt token tables
    tie_word_embeddings: bool = True  # tie LM head to target embedding

    # Classification fine-tuning head (encoder-only consumption).
    num_labels: int = 2
    classifier_dropout: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_decoder_layers <= 0:
            raise ValueError(
                f"num_decoder_layers must be positive, got {self.num_decoder_layers}"
            )
        if self.decoder_vocab_size is not None and self.decoder_vocab_size <= 0:
            raise ValueError(
                f"decoder_vocab_size must be positive when set, got {self.decoder_vocab_size}"
            )
        if self.share_embeddings:
            tgt_v = self.decoder_vocab_size or self.vocab_size
            if tgt_v != self.vocab_size:
                raise ValueError(
                    "share_embeddings=True requires decoder_vocab_size == vocab_size "
                    f"(or None), got src={self.vocab_size} / tgt={tgt_v}"
                )
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.classifier_dropout is not None and not (
            0.0 <= self.classifier_dropout < 1.0
        ):
            raise ValueError(
                "classifier_dropout must be in [0, 1) or None, got "
                f"{self.classifier_dropout}"
            )

    @property
    def effective_decoder_vocab_size(self) -> int:
        """Resolved target-side vocabulary (handles ``decoder_vocab_size=None``)."""
        return (
            self.decoder_vocab_size
            if self.decoder_vocab_size is not None
            else self.vocab_size
        )
