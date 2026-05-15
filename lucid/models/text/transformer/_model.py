"""Original Transformer model (Vaswani et al., 2017) — encoder-decoder seq2seq.

The bare paper architecture — six-layer encoder + six-layer decoder, sinusoidal
positional encoding, multi-head attention, position-wise FFN.  Public API
mirrors the rest of :mod:`lucid.models.text` (BERT / GPT / GPT-2 / RoFormer)
so callers move between families with no extra ceremony.

Task heads
----------
    TransformerModel                     — bare encoder-decoder trunk
    TransformerForSeq2SeqLM              — translation / summarisation head
    TransformerForSequenceClassification — encoder-only sentence classifier
    TransformerForTokenClassification    — encoder-only per-token classifier

Generation
----------
``TransformerForSeq2SeqLM.generate`` performs greedy / sampled decoding by
running the encoder once and unrolling the decoder one token at a time.  It
does **not** reuse :class:`lucid.models.GenerationMixin` because that mixin
is decoder-only — encoder-decoder semantics differ enough to warrant a
local implementation.
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import MaskedLMMixin
from lucid.models._output import (
    MaskedLMOutput,
    Seq2SeqLMOutput,
)
from lucid.models.text.transformer._config import TransformerConfig

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _causal_mask(T: int, device: str) -> Tensor:
    """``(T, T)`` additive mask: ``0`` on / below the diagonal, ``-1e4`` above.

    Plugs into :class:`lucid.nn.Transformer` as ``tgt_mask`` so the decoder
    cannot attend to future positions during training.
    """
    base = lucid.tril(lucid.ones((T, T), device=device))
    return (1.0 - base) * -1e4


def _key_padding_to_kpm(mask: Tensor | None) -> Tensor | None:
    """Convert HF-style 0/1 padding mask to ``nn.Transformer`` key_padding_mask.

    HF convention: ``1`` = attend, ``0`` = ignore.
    nn.Transformer convention: ``True`` = ignore, ``False`` = attend.
    """
    if mask is None:
        return None
    return mask.float() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bare encoder-decoder trunk
# ─────────────────────────────────────────────────────────────────────────────


class TransformerModel(PretrainedModel):
    """Vaswani et al. (2017) seq2seq encoder-decoder trunk.

    Forward inputs / outputs follow the :class:`lucid.models.Seq2SeqLMOutput`
    contract.  For task-specific outputs use the ``TransformerForXxx`` wrappers.
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._max_pos = config.max_position_embeddings
        self._d_model = config.hidden_size
        tgt_vocab = config.effective_decoder_vocab_size

        # Token embeddings.  When ``share_embeddings`` is True the two tables
        # are the same parameter (saves memory, requires identical vocabs).
        self.src_tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        if config.share_embeddings:
            self.tgt_tok_emb = self.src_tok_emb
        else:
            self.tgt_tok_emb = nn.Embedding(tgt_vocab, config.hidden_size)

        # Sinusoidal positional encoding from :mod:`lucid.nn`.
        self.positional_encoding = nn.SinusoidalEmbedding(
            num_positions=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout)

        # Activation choice mapped to nn.Transformer's allowed set.
        act = "gelu" if config.hidden_act in ("gelu", "gelu_new") else "relu"
        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            num_encoder_layers=config.num_hidden_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout,
            activation=act,
            batch_first=True,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.src_tok_emb

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"TransformerModel input embeddings must be nn.Embedding, got {type(value).__name__}"
            )
        self.src_tok_emb = value

    def get_output_embeddings(self) -> nn.Module:
        """Target-side embedding — used to tie an LM head against."""
        return self.tgt_tok_emb

    def _embed(self, ids: Tensor, table: nn.Module) -> Tensor:
        """Token-embed + scale-by-√d + sin/cos PE + dropout."""
        T = int(ids.shape[1])
        if T > self._max_pos:
            raise ValueError(
                f"Sequence length {T} exceeds max_position_embeddings {self._max_pos}"
            )
        emb = cast(Tensor, table(ids)) * math.sqrt(self._d_model)
        pe = cast(Tensor, self.positional_encoding())[:T]  # (T, d_model)
        emb = emb + pe.unsqueeze(0)  # broadcast over batch
        return cast(Tensor, self.dropout(emb))

    def encode(
        self,
        src_ids: Tensor,
        src_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run only the encoder; returns memory ``(B, S, d_model)``."""
        src_emb = self._embed(src_ids, self.src_tok_emb)
        kpm = _key_padding_to_kpm(src_attention_mask)
        memory = cast(
            Tensor, self.transformer.encoder(src_emb, src_key_padding_mask=kpm)
        )
        return memory

    def decode(
        self,
        tgt_ids: Tensor,
        memory: Tensor,
        tgt_attention_mask: Tensor | None = None,
        memory_attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run only the decoder against precomputed encoder memory."""
        T = int(tgt_ids.shape[1])
        dev = tgt_ids.device.type
        tgt_emb = self._embed(tgt_ids, self.tgt_tok_emb)
        tgt_mask = _causal_mask(T, device=dev)
        tgt_kpm = _key_padding_to_kpm(tgt_attention_mask)
        mem_kpm = _key_padding_to_kpm(memory_attention_mask)
        return cast(
            Tensor,
            self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=mem_kpm,
            ),
        )

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """Full encode + decode pass.

        Args:
            input_ids:            ``(B, S)`` source token ids.
            decoder_input_ids:    ``(B, T)`` target token ids (right-shifted).
            attention_mask:       Optional ``(B, S)`` HF-style padding mask.
            decoder_attention_mask: Optional ``(B, T)`` HF-style padding mask.

        Returns:
            :class:`Seq2SeqLMOutput` carrying decoder hidden states as
            ``logits`` placeholder (no head yet) plus encoder memory.  The
            ``ForSeq2SeqLM`` wrapper above this projects into the LM head.
        """
        memory = self.encode(input_ids, attention_mask)
        decoded = self.decode(
            decoder_input_ids,
            memory,
            tgt_attention_mask=decoder_attention_mask,
            memory_attention_mask=attention_mask,
        )
        return Seq2SeqLMOutput(
            logits=decoded,
            encoder_last_hidden_state=memory,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Seq2SeqLM head — translation / summarisation
# ─────────────────────────────────────────────────────────────────────────────


class TransformerForSeq2SeqLM(PretrainedModel):
    """Encoder-decoder Transformer + LM head (tied to target embeddings).

    Standard translation / summarisation interface — pass ``input_ids`` (the
    source) and ``decoder_input_ids`` (the right-shifted target); supply
    ``labels`` to also compute the shift-loss for training.
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerModel(config)
        tgt_vocab = config.effective_decoder_vocab_size
        self.lm_head = nn.Linear(config.hidden_size, tgt_vocab, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.transformer.tgt_tok_emb.weight

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        decoder_input_ids: Tensor,
        attention_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        outputs = cast(
            Seq2SeqLMOutput,
            self.transformer(
                input_ids,
                decoder_input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
            ),
        )
        logits = cast(Tensor, self.lm_head(outputs.logits))  # decoder hidden → vocab

        loss: Tensor | None = None
        if labels is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                labels.reshape(B * T).long(),
                ignore_index=-100,
            )

        return Seq2SeqLMOutput(
            logits=logits,
            loss=loss,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        )

    @lucid.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_length: int = 32,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Greedy seq2seq decoding.

        Encodes ``input_ids`` once, then unrolls the decoder one greedy
        argmax token at a time until ``max_length`` (or every batch row hits
        ``eos_token_id``).  Encoder-decoder generation deliberately doesn't
        reuse :class:`GenerationMixin` because that mixin is decoder-only.

        Args:
            input_ids:     ``(B, S)`` source token ids.
            max_length:    Cap on generated target length (incl. BOS).
            bos_token_id:  Start-of-sequence id; defaults to
                           ``config.bos_token_id`` then 0.
            eos_token_id:  End-of-sequence id; defaults to
                           ``config.eos_token_id``.
            pad_token_id:  Right-padding for finished rows; defaults to
                           ``config.pad_token_id`` then 0.
            attention_mask: Optional source padding mask.

        Returns:
            ``(B, T_out)`` int Tensor of generated target tokens.
        """
        cfg = cast(TransformerConfig, self.config)
        bos = bos_token_id if bos_token_id is not None else cfg.bos_token_id
        eos = eos_token_id if eos_token_id is not None else cfg.eos_token_id
        pad = pad_token_id if pad_token_id is not None else cfg.pad_token_id
        if bos is None:
            bos = 0
        if pad is None:
            pad = 0

        B = int(input_ids.shape[0])
        dev = input_ids.device.type
        memory = self.transformer.encode(input_ids, attention_mask)

        finished: list[bool] = [False] * B
        out_tokens: list[Tensor] = [lucid.tensor([bos] * B, device=dev).long()]

        for _ in range(max_length - 1):
            tgt = lucid.stack(out_tokens, dim=1)  # (B, T_cur)
            decoded = self.transformer.decode(
                tgt, memory, memory_attention_mask=attention_mask
            )  # (B, T_cur, d_model)
            next_logits = cast(Tensor, self.lm_head(decoded[:, -1, :]))  # (B, V)
            next_tok = lucid.argmax(next_logits, dim=-1)  # (B,)

            row: list[int] = [int(next_tok[b].item()) for b in range(B)]
            for b in range(B):
                if finished[b]:
                    row[b] = pad
                elif eos is not None and row[b] == eos:
                    finished[b] = True
            out_tokens.append(lucid.tensor(row, device=dev).long())

            if all(finished):
                break

        return lucid.stack(out_tokens, dim=1).long()


# ─────────────────────────────────────────────────────────────────────────────
# Encoder-only downstream task wrappers
# ─────────────────────────────────────────────────────────────────────────────


class TransformerForSequenceClassification(PretrainedModel):
    """Encoder-only sentence classifier — pools the first source-side token.

    Pattern mirrors :class:`BertForSequenceClassification`; the decoder half
    of the transformer is unused so this head trains the same trunk
    efficiently for GLUE-style downstream tasks.
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        memory = self.transformer.encode(input_ids, attention_mask)  # (B, S, d)
        pooled = memory[:, 0]  # (B, d)
        pooled = cast(Tensor, self.dropout(pooled))
        logits = cast(Tensor, self.classifier(pooled))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return MaskedLMOutput(logits=logits, loss=loss)


class TransformerForTokenClassification(PretrainedModel, MaskedLMMixin):
    """Encoder-only per-token classifier — NER / POS heads.

    Uses :class:`MaskedLMMixin` for the standard ``(B, T, V) → loss`` reduction
    that BERT / RoFormer also share.
    """

    config_class: ClassVar[type[TransformerConfig]] = TransformerConfig
    base_model_prefix: ClassVar[str] = "transformer"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.transformer = TransformerModel(config)
        drop = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout
        )
        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(  # type: ignore[override]
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> MaskedLMOutput:
        memory = self.transformer.encode(input_ids, attention_mask)  # (B, S, d)
        seq = cast(Tensor, self.dropout(memory))
        logits = cast(Tensor, self.classifier(seq))

        loss: Tensor | None = None
        if labels is not None:
            loss = self.compute_lm_loss(logits, labels)

        return MaskedLMOutput(logits=logits, loss=loss)
