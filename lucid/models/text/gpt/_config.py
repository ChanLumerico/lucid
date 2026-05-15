"""GPT-1 configuration (Radford et al., 2018).

Paper: "Improving Language Understanding by Generative Pre-Training".  Single
canonical size in the original paper (L=12, H=768, A=12, T=512); we expose a
``GPTConfig`` parametrised the same way as ``BertConfig`` so downstream code
can build smaller variants for tests / quick experiments without monkey-
patching.

Differences from BERT:
    - decoder-only (causal mask applied inside attention)
    - no segment embedding (``type_vocab_size`` is unused)
    - default activation is ``gelu_new`` (Hendrycks tanh approximation, per the
      GPT-1 reference code)
    - tied input/output embeddings by default
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models.text._config import LanguageModelConfig, TextActivation


@dataclass(frozen=True)
class GPTConfig(LanguageModelConfig):
    """Configuration for every GPT-1 variant.

    Args mirror the HuggingFace ``OpenAIGPTConfig`` field set so checkpoint
    porting is a flat key rename.
    """

    model_type: ClassVar[str] = "gpt"

    # GPT-1 defaults (Radford et al., 2018).
    vocab_size: int = 40_478  # BPE vocab from the paper
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072
    max_position_embeddings: int = 512
    hidden_act: TextActivation = "gelu_new"
    pad_token_id: int | None = None  # GPT-1 has no pad token

    # Classification fine-tuning head — used by ``GPTForSequenceClassification``.
    num_labels: int = 2
    classifier_dropout: float | None = None  # falls back to ``hidden_dropout``

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.classifier_dropout is not None and not (
            0.0 <= self.classifier_dropout < 1.0
        ):
            raise ValueError(
                "classifier_dropout must be in [0, 1) or None, got "
                f"{self.classifier_dropout}"
            )
