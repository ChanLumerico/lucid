"""HF BERT checkpoints → Lucid ``bert_*`` weights.

BERT is a *pure identity map*: Lucid mirrors the upstream ``BertModel``
parameter naming one-for-one (verified: 199/199 keys, same names + shapes
for ``bert-base-uncased``), so the base encoders need no key renames and no
value transforms.

Two checkpoint families are covered:

* **Base encoders** (``bert_tiny`` … ``bert_large``) — the bare ``BertModel``
  (embeddings + encoder + pooler).  Sourced from ``AutoModel``.  The four
  miniatures are the Turc et al. 2019 "Well-Read Students" pre-distilled
  sizes; ``base`` / ``large`` are Devlin et al. 2018.
* **Masked-LM heads** (``bert_base_mlm`` / ``bert_large_mlm``) — base encoder
  + ``cls.predictions`` head.  Sourced from ``BertForPreTraining`` (so the
  pooler weights, which Lucid keeps but ``BertForMaskedLM`` drops, come from
  the real checkpoint).  Two upstream redundancies are dropped: the NSP head
  (``cls.seq_relationship.*``) and the duplicate ``cls.predictions.decoder.bias``
  (a tied alias of ``cls.predictions.bias``, which Lucid stores once).
"""

import dataclasses
import enum
from collections.abc import Callable

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    "Devlin et al., \"BERT: Pre-training of Deep Bidirectional Transformers "
    "for Language Understanding\", NAACL 2019. Miniatures: Turc et al., "
    "\"Well-Read Students Learn Better\", 2019."
)
_PAPER_URL = "https://arxiv.org/abs/1810.04805"

# arch_key -> (lucid_factory, repo_slug, title, hf_model_id, kind)
_VARIANTS: dict[str, tuple[str, str, str, str, str]] = {
    "bert_tiny": (
        "bert_tiny", "bert-tiny", "BERT-Tiny",
        "google/bert_uncased_L-2_H-128_A-2", "base",
    ),
    "bert_mini": (
        "bert_mini", "bert-mini", "BERT-Mini",
        "google/bert_uncased_L-4_H-256_A-4", "base",
    ),
    "bert_small": (
        "bert_small", "bert-small", "BERT-Small",
        "google/bert_uncased_L-4_H-512_A-8", "base",
    ),
    "bert_medium": (
        "bert_medium", "bert-medium", "BERT-Medium",
        "google/bert_uncased_L-8_H-512_A-8", "base",
    ),
    "bert_base": (
        "bert_base", "bert-base", "BERT-Base",
        "google-bert/bert-base-uncased", "base",
    ),
    "bert_large": (
        "bert_large", "bert-large", "BERT-Large",
        "google-bert/bert-large-uncased", "base",
    ),
    "bert_base_mlm": (
        "bert_base_mlm", "bert-base-mlm", "BERT-Base (Masked-LM)",
        "google-bert/bert-base-uncased", "mlm",
    ),
    "bert_large_mlm": (
        "bert_large_mlm", "bert-large-mlm", "BERT-Large (Masked-LM)",
        "google-bert/bert-large-uncased", "mlm",
    ),
}


def _jsonable(value: object) -> object:
    """Make a config field JSON-serialisable (enum → its value, tuple → list)."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class BertArch(Architecture):
    """Identity-map converter for the BERT family (base + masked-LM)."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"BertArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, hf_id, kind = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._hf_id = hf_id
        self._kind = kind

        if kind == "mlm":
            from transformers import BertForPreTraining

            self._src = BertForPreTraining.from_pretrained(hf_id).eval()
        else:
            from transformers import AutoModel

            self._src = AutoModel.from_pretrained(hf_id).eval()

        import lucid.models as models

        self._model: Module = getattr(models, factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {
            k: v.detach().cpu().numpy() for k, v in self._src.state_dict().items()
        }

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        if self._kind == "mlm":
            # NSP head is unused by Lucid's masked-LM model.
            if src_key.startswith("cls.seq_relationship."):
                return None
            # HF stores the decoder bias twice (it is tied to predictions.bias);
            # Lucid keeps only the standalone `cls.predictions.bias`.
            if src_key == "cls.predictions.decoder.bias":
                return None
        return src_key

    def spec(self) -> ConversionSpec:
        cfg = self._model.config
        config = {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}
        task = "masked-lm" if self._kind == "mlm" else "base"
        num_classes = cfg.vocab_size if self._kind == "mlm" else cfg.hidden_size

        preprocessing = {
            "tokenizer_class": "BertTokenizer",
            "vocab_size": cfg.vocab_size,
            "do_lower_case": True,
            "max_length": cfg.max_position_embeddings,
        }
        meta: dict[str, object] = {
            "num_params": int(sum(p.numel() for p in self._src.parameters())),
            "recipe": f"HuggingFace/{self._hf_id}",
            "metrics": {},
        }
        return ConversionSpec(
            model_name=self._factory,
            architecture=self.arch,
            repo_id=f"lucid-dl/{self._slug}",
            tag=self.tag,
            task=task,
            model_type="bert",
            source=f"transformers/{self._hf_id}",
            license="apache-2.0",
            num_classes=num_classes,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=["wikipedia", "bookcorpus"],
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return BertArch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
