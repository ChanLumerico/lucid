"""BERT configuration (Devlin et al., 2018).

Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding".  Field set mirrors the HuggingFace ``BertConfig`` so existing
tokenizers / checkpoints map 1-to-1 against Lucid weights.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models.text._config import LanguageModelConfig


@dataclass(frozen=True)
class BertConfig(LanguageModelConfig):
    """Configuration for every BERT variant.

    Inherits all common text fields from :class:`LanguageModelConfig` and adds
    the two knobs unique to BERT:

    Args:
        type_vocab_size: Size of the segment-id vocabulary used by the
            next-sentence objective.  ``2`` for vanilla BERT (segment A / B);
            single-segment fine-tunes still use the full table but only ever
            feed zeros.
        position_embedding_type: Position encoding flavour. ``"absolute"`` is
            the only one shipped today — relative-position variants
            (T5-style, RoFormer-style) live in their own families.
    """

    model_type: ClassVar[str] = "bert"

    # BERT defaults match ``bert-base-uncased`` for vocab + layer counts.
    vocab_size: int = 30_522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 512
    pad_token_id: int | None = 0

    # BERT-specific
    type_vocab_size: int = 2
    position_embedding_type: Literal["absolute"] = "absolute"

    # Downstream task knobs (used by the ``For*`` heads).
    num_labels: int = 2
    classifier_dropout: float | None = None  # falls back to ``hidden_dropout``

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type_vocab_size <= 0:
            raise ValueError(
                f"type_vocab_size must be positive, got {self.type_vocab_size}"
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
