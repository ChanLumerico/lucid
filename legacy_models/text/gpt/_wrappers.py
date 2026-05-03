import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid_legacy.models.base import PreTrainedModelMixin


from ._model import GPT, GPTConfig

__all__ = ["GPTLMHeadModel", "GPTDoubleHeadsModel", "GPTForSequenceClassification"]


class _GPTTaskWrapperMixin:
    gpt: GPT

    def get_input_embeddings(self) -> nn.Embedding:
        return self.gpt.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.gpt.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear | None:
        return getattr(self, "lm_head", None)

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self) -> None:
        out = self.get_output_embeddings()
        if out is not None:
            out.weight = self.get_input_embeddings().weight


def _causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def _classification_ce_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


class GPTLMHeadModel(_GPTTaskWrapperMixin, PreTrainedModelMixin, nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.gpt = GPT(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list[nn.KVCache] | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor | None, Tensor, list[nn.KVCache] | None]:
        hidden_states, presents = self.gpt(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = _causal_lm_loss(logits, labels) if labels is not None else None
        return loss, logits, presents


class _GPTMultipleChoiceHead(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.summary = nn.Linear(config.hidden_size, 1)
        self.drop = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, mc_token_ids: Tensor) -> Tensor:
        N, C, L, H = hidden_states.shape
        idx = mc_token_ids.unsqueeze(axis=-1).unsqueeze(axis=-1).expand(N, C, 1, H)
        mc_hidden = hidden_states.gather(axis=2, index=idx).squeeze(axis=2)

        return self.summary(self.drop(mc_hidden)).squeeze(axis=-1)


class GPTDoubleHeadsModel(_GPTTaskWrapperMixin, PreTrainedModelMixin, nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.gpt = GPT(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mc_head = _GPTMultipleChoiceHead(config)
        self.tie_weights()

    def forward(
        self,
        input_ids: Tensor,
        mc_token_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        mc_labels: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor, Tensor]:
        N, C, L = input_ids.shape

        flat_input_ids = input_ids.reshape(N * C, L)
        flat_attn_mask = (
            attention_mask.reshape(N * C, L) if attention_mask is not None else None
        )
        flat_position_ids = (
            position_ids.reshape(N * C, L) if position_ids is not None else None
        )

        hidden_states, presents = self.gpt(
            flat_input_ids,
            attention_mask=flat_attn_mask,
            position_ids=flat_position_ids,
            use_cache=use_cache,
        )

        lm_logits = self.lm_head(hidden_states).reshape(N, C, L, -1)
        mc_logits = self.mc_head(hidden_states.reshape(N, C, L, -1), mc_token_ids)

        lm_loss, mc_loss = None, None
        if labels is not None:
            flat_logits = lm_logits.reshape(N * C, L, -1)
            flat_labels = labels.reshape(N * C, L)
            lm_loss = _causal_lm_loss(flat_logits, flat_labels)

        if mc_labels is not None:
            mc_loss = F.cross_entropy(mc_logits, mc_labels)

        return lm_loss, mc_loss, lm_logits, mc_logits, presents


class GPTForSequenceClassification(
    _GPTTaskWrapperMixin, PreTrainedModelMixin, nn.Module
):
    def __init__(self, config: GPTConfig, num_labels: int = 2) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        self.gpt = GPT(config)
        self.score = nn.Linear(config.hidden_size, num_labels, bias=False)
        self.apply(self.gpt._init_weights)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor]:
        hidden_states, _ = self.gpt(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        if self.config.pad_token_id is None:
            seq_lengths = -1
        else:
            seq_lengths = (input_ids != self.config.pad_token_id).sum(axis=-1) - 1

        pooled = hidden_states[range(hidden_states.shape[0]), seq_lengths]
        logits = self.score(pooled)

        loss = _classification_ce_loss(logits, labels) if labels is not None else None
        return loss, logits
