"""RoFormer configuration (Su et al., 2021).

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding".
RoFormer is BERT with absolute position embeddings replaced by rotary
position embedding (RoPE) inside every attention layer.  The config
therefore mirrors BERT's, minus the absolute ``position_embedding_type``
and plus the RoPE-specific ``rotary_base``.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models.text._config import LanguageModelConfig


@dataclass(frozen=True)
class RoFormerConfig(LanguageModelConfig):
    """Configuration for every RoFormer variant.

    Args:
        rotary_base: Frequency base ``θ_0`` for the rotary embedding
            (``θ_i = base ** (-2 i / d_head)``).  10000.0 per the paper.
        type_vocab_size: Segment-id vocabulary (kept for BERT-parity even
            though RoFormer fine-tunes typically feed a single segment).
        position_embedding_type: Always ``"rotary"`` here — kept as a literal
            for forward compat with future variants (NTK-aware scaling, etc.).
        num_labels / classifier_dropout: Downstream classification head knobs.
    """

    model_type: ClassVar[str] = "roformer"

    # RoFormer-base in the paper: L=12, H=768, A=12, intermediate=3072.
    vocab_size: int = 50_000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 1_536  # paper extends BERT's 512
    pad_token_id: int | None = 0

    # RoPE-specific.
    rotary_base: float = 10_000.0
    position_embedding_type: Literal["rotary"] = "rotary"

    # Inherited-from-BERT extras.
    type_vocab_size: int = 2
    num_labels: int = 2
    classifier_dropout: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"RoFormer requires an even head_dim (hidden_size / num_attention_heads); "
                f"got head_dim={head_dim}."
            )
        if self.rotary_base <= 1.0:
            raise ValueError(
                f"rotary_base must be > 1 for RoPE to make sense, got {self.rotary_base}"
            )
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
