"""Model output dataclasses — iterable, dict-like, tuple-compatible.

All ``forward()`` methods on :class:`PretrainedModel` subclasses return one
of these.  Each output behaves both as a dataclass with named fields and
as a tuple/sequence (skipping ``None`` fields), so callers can pick either
style without a wrapper.
"""

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class ModelOutput:
    """Base class for all model outputs.

    Subclasses must use ``@dataclass`` so :func:`dataclasses.fields` works.
    Direct instantiation of ``ModelOutput`` itself is meaningless (no fields)
    and should be avoided.
    """

    def __iter__(self) -> Iterator[Tensor]:
        for f in fields(self):  # type: ignore[arg-type]
            value: Tensor | None = getattr(self, f.name)
            if value is not None:
                yield value

    def __getitem__(self, idx: int | str) -> Tensor:
        if isinstance(idx, str):
            value2: Tensor | None = getattr(self, idx, None)
            if value2 is None:
                raise KeyError(idx)
            return value2
        return tuple(self)[idx]

    def __len__(self) -> int:
        count = 0
        for f in fields(self):  # type: ignore[arg-type]
            if getattr(self, f.name) is not None:
                count += 1
        return count

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return getattr(self, key, None) is not None

    def to_tuple(self) -> tuple[Tensor, ...]:
        """Tuple of non-None field values, in declaration order."""
        return tuple(self)

    def keys(self) -> list[str]:
        """Names of fields whose value is not None, in declaration order."""
        return [
            f.name for f in fields(self)  # type: ignore[arg-type]
            if getattr(self, f.name) is not None
        ]

    def values(self) -> list[Tensor]:
        return list(self)

    def items(self) -> list[tuple[str, Tensor]]:
        return [(k, getattr(self, k)) for k in self.keys()]


@dataclass
class BaseModelOutput(ModelOutput):
    """Generic backbone forward output."""

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    """Backbone output + pooled (e.g. CLS) representation."""

    last_hidden_state: Tensor
    pooler_output: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class ImageClassificationOutput(ModelOutput):
    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class ObjectDetectionOutput(ModelOutput):
    logits: Tensor
    pred_boxes: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass
class SemanticSegmentationOutput(ModelOutput):
    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass
class CausalLMOutput(ModelOutput):
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class MaskedLMOutput(ModelOutput):
    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    decoder_hidden_states: tuple[Tensor, ...] | None = None
    decoder_attentions: tuple[Tensor, ...] | None = None
    encoder_last_hidden_state: Tensor | None = None
    encoder_hidden_states: tuple[Tensor, ...] | None = None
    encoder_attentions: tuple[Tensor, ...] | None = None
