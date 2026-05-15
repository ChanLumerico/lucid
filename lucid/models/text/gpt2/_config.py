"""GPT-2 configuration (Radford et al., 2019).

Paper: "Language Models are Unsupervised Multitask Learners".  Key
differences from GPT-1 captured in the model file (pre-LN layout, final
``ln_f``, larger BPE vocab, longer context).  This config only stores the
knobs — see :mod:`lucid.models.text.gpt2._model` for the architectural
choices that follow from them.
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models.text._config import LanguageModelConfig, TextActivation


@dataclass(frozen=True)
class GPT2Config(LanguageModelConfig):
    """Configuration for every GPT-2 variant.

    Defaults match the **124M-parameter** "small" checkpoint released in 2019
    (the file historically known as ``117M``).  Larger sizes are reached by
    overriding ``hidden_size`` / ``num_hidden_layers`` / ``num_attention_heads``
    — see :mod:`lucid.models.text.gpt2._pretrained` for the canonical table.
    """

    model_type: ClassVar[str] = "gpt2"

    # GPT-2 small (124M) defaults.
    vocab_size: int = 50_257  # BPE merges from the official release
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3_072  # 4 × hidden_size
    max_position_embeddings: int = 1_024
    hidden_act: TextActivation = "gelu_new"

    # The released BPE tokenizer assigns the EOT token (``<|endoftext|>``) id
    # 50256 — used as both BOS and EOS, with no padding token.
    pad_token_id: int | None = None
    bos_token_id: int | None = 50_256
    eos_token_id: int | None = 50_256

    layer_norm_eps: float = 1e-5  # HF default for GPT-2 (vs 1e-12 in BERT)

    # Classification fine-tuning head.
    num_labels: int = 2
    classifier_dropout: float | None = None  # falls back to ``hidden_dropout``

    # GPT-2 scales the residual projection's init by ``1 / sqrt(2N)`` where
    # N is the number of layers.  Stored here so the model class can pick it
    # up without re-deriving from ``num_hidden_layers``.
    scale_residual_init: bool = True

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
