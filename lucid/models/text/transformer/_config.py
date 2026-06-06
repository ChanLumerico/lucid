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
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.text._config import LanguageModelConfig


@model_family_meta(
    canonical_name="Transformer",
    citation=(
        'Vaswani, Ashish, et al. "Attention Is All You Need." '
        "Advances in Neural Information Processing Systems, 2017, "
        "pp. 5998–6008."
    ),
    theory=r"""
    The Transformer replaces the recurrence of RNN-based seq2seq models
    with **scaled dot-product attention**, enabling fully parallel
    sequence processing.  For query/key/value matrices
    :math:`Q \in \mathbb{R}^{T_q \times d_k}`,
    :math:`K \in \mathbb{R}^{T_k \times d_k}`, and
    :math:`V \in \mathbb{R}^{T_k \times d_v}`, attention is

    .. math::

        \operatorname{Attention}(Q, K, V)
            = \operatorname{softmax}\!\left(
                \frac{Q K^\top}{\sqrt{d_k}}
              \right) V,

    where the :math:`1/\sqrt{d_k}` factor counteracts the variance of
    dot products in high dimensions and keeps softmax gradients
    well-conditioned.

    **Multi-head attention** runs :math:`h` parallel attention heads on
    learned projections of :math:`(Q, K, V)` into :math:`d_k = d_v =
    d_{\text{model}} / h` subspaces, concatenates the outputs, and
    applies a final linear:

    .. math::

        \operatorname{MultiHead}(Q, K, V) =
            \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O,
        \qquad
        \text{head}_i = \operatorname{Attention}(Q W_i^Q, K W_i^K, V W_i^V).

    Each encoder layer stacks multi-head self-attention and a
    position-wise feed-forward network (with hidden width
    :math:`d_{\text{ff}}=2048`), wrapped in residual connections and
    post-LayerNorm.  Each decoder layer adds **causal self-attention**
    over previously generated targets and **cross-attention** over the
    encoder output.

    Position information is injected via **sinusoidal positional
    encodings** of fixed (untrained) frequencies, which generalise to
    sequence lengths unseen during training.  The paper's *base* model
    (:math:`N=6` encoder/decoder layers, :math:`d_{\text{model}}=512`,
    :math:`h=8`) achieves state-of-the-art WMT 2014 En-De / En-Fr BLEU
    while training in a fraction of the time of prior recurrent and
    convolutional seq2seq systems — the architectural foundation for
    essentially every modern language model.
    """,
)
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

    @override
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
