"""Composable mixins for model families.

Tier-2 reusable behaviour — added only when ≥3 families would otherwise
duplicate the same logic.  Mixins carry no state (no ``__init__`` of
their own); they expose methods that operate on attributes the host class
is contracted to provide.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor.tensor import Tensor

if TYPE_CHECKING:
    from lucid.models._output import GenerationOutput
    from lucid.models.generative._schedulers import Scheduler


@dataclass(frozen=True)
class FeatureInfo:
    """Output spec for one stage of a backbone (timm-compatible)."""

    stage: int
    num_channels: int
    reduction: int  # spatial down-sampling factor vs the input


class BackboneMixin(ABC):
    """Marker mixin: a model that can serve as a feature extractor.

    Subclasses must implement :meth:`forward_features` returning the
    deepest stage's feature map and :attr:`feature_info` enumerating
    every emitted stage's channel / stride spec.
    """

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor: ...

    @property
    @abstractmethod
    def feature_info(self) -> list[FeatureInfo]: ...


class ClassificationHeadMixin:
    """Standard ``classifier`` Linear head + transfer-learning hook.

    Subclasses must call :meth:`_build_classifier` in their ``__init__``
    *after* ``super().__init__(config)`` to install the head.  Use
    :meth:`reset_classifier` to swap ``num_classes`` post hoc.
    """

    classifier: nn.Module

    def _build_classifier(
        self,
        in_features: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        if dropout > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the final Linear with a freshly initialised one."""
        if isinstance(self.classifier, nn.Linear):
            in_features = int(self.classifier.in_features)
            self.classifier = nn.Linear(in_features, num_classes)
            return
        if isinstance(self.classifier, nn.Sequential):
            for i in range(len(self.classifier) - 1, -1, -1):
                m = self.classifier[i]
                if isinstance(m, nn.Linear):
                    in_features = int(m.in_features)
                    self.classifier[i] = nn.Linear(in_features, num_classes)
                    return
            raise RuntimeError(
                "reset_classifier: no Linear layer found in classifier Sequential"
            )
        raise NotImplementedError(
            f"reset_classifier not implemented for " f"{type(self.classifier).__name__}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MaskedLMMixin — per-token cross-entropy loss for masked / token-level heads
# ─────────────────────────────────────────────────────────────────────────────


class MaskedLMMixin:
    """Per-token cross-entropy loss helper for masked-LM / token-classification
    heads.

    Encoder text families (BERT, RoFormer, …) all reduce a ``(B, T, V)`` logit
    tensor against ``(B, T)`` labels with the same recipe: flatten the
    sequence axis into the batch axis, then run
    :func:`lucid.nn.functional.cross_entropy` with ``ignore_index=-100``.
    Centralising the call here removes a 3-line stanza repeated across every
    ``ForMaskedLM`` / ``ForTokenClassification`` class.
    """

    @staticmethod
    def compute_lm_loss(
        logits: Tensor,
        labels: Tensor,
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        """Flatten ``(B, T, V)`` logits against ``(B, T)`` labels and reduce
        with cross-entropy.

        Args:
            logits:       ``(B, T, V)`` per-token logits.
            labels:       ``(B, T)`` int target ids.  Entries equal to
                          ``ignore_index`` contribute zero loss / gradient.
            ignore_index: Token id to skip (HuggingFace convention: ``-100``).

        Returns:
            Scalar loss tensor.
        """
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T).long(),
            ignore_index=ignore_index,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GenerationMixin — autoregressive text sampling
# ─────────────────────────────────────────────────────────────────────────────


class GenerationMixin:
    """Decoder-only autoregressive sampling for causal LM heads.

    Concrete subclasses must:

    * Inherit from :class:`lucid.models.PretrainedModel`.
    * Have a ``forward(input_ids, ...)`` that returns a
      :class:`lucid.models.CausalLMOutput` with ``logits`` shaped
      ``(B, T, vocab_size)``.
    * Expose a ``config`` attribute carrying ``vocab_size`` /
      ``pad_token_id`` / ``bos_token_id`` / ``eos_token_id`` (the fields
      :class:`lucid.models.text.LanguageModelConfig` defines).

    The mixin then provides :meth:`generate` — greedy or stochastic
    sampling with ``temperature``, ``top_k``, ``top_p``, and
    ``repetition_penalty`` knobs.  Sampling stops per-sequence when
    ``eos_token_id`` is produced (or unconditionally at ``max_length``).

    Notes on cache
    --------------
    The first implementation does **not** use the model's KV cache —
    every step re-runs the full prefix.  This is correct but O(T²) in
    the prefix length.  Plan to add cache support when GPT-2 lands and
    we have a concrete ``past_key_values`` shape to plumb through.
    """

    @lucid.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_length: int = 20,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressively extend ``input_ids`` until a stop condition.

        Args:
            input_ids: ``(B, T_prompt)`` int prompt tokens.
            max_length: Total sequence length cap (prompt + generated).
                Ignored when ``max_new_tokens`` is supplied.
            max_new_tokens: Cap on tokens to generate, additive over the
                prompt length.  Takes precedence over ``max_length``.
            do_sample: ``False`` → greedy argmax; ``True`` → temperature
                / top-k / top-p sampling.
            temperature: Logit divisor before softmax.  ``< 1`` sharpens,
                ``> 1`` flattens.  Ignored under greedy decoding.
            top_k: Keep only the K highest-probability tokens before
                sampling.  ``None`` disables.
            top_p: Nucleus sampling — keep the smallest token set whose
                cumulative probability ≥ ``top_p``.  ``None`` disables.
            repetition_penalty: Multiply the logit of every previously
                generated token by ``1 / penalty`` (HuggingFace
                convention).  ``1.0`` → no effect.
            pad_token_id: Token id used to pad finished sequences after
                they emit ``eos_token_id``.  Defaults to
                ``config.pad_token_id`` then 0.
            eos_token_id: Stop generating per-sequence once this id is
                emitted.  Defaults to ``config.eos_token_id``.

        Returns:
            ``(B, T_final)`` int Tensor where ``T_final`` ≤ ``max_length``
            (or ``T_prompt + max_new_tokens``).  Sequences that hit
            ``eos_token_id`` early are right-padded with ``pad_token_id``.
        """
        # ── Resolve token id defaults from the host model's config ────────
        cfg = getattr(self, "config", None)
        if eos_token_id is None and cfg is not None:
            eos_token_id = getattr(cfg, "eos_token_id", None)
        if pad_token_id is None and cfg is not None:
            pad_token_id = getattr(cfg, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        if input_ids.ndim != 2:
            raise ValueError(
                f"generate(): input_ids must be 2-D (B, T), got shape {tuple(input_ids.shape)}"
            )

        B = int(input_ids.shape[0])
        T_prompt = int(input_ids.shape[1])

        if max_new_tokens is not None:
            stop_len = T_prompt + int(max_new_tokens)
        else:
            stop_len = int(max_length)
        if stop_len <= T_prompt:
            return input_ids

        dev = input_ids.device.type
        # Per-row "is finished" flag — once True, future tokens are pad.
        finished: list[bool] = [False] * B
        # We grow a Python list of (B,) int rows then stack at the end so
        # we don't repeatedly re-allocate the full prefix tensor.
        out_tokens: list[Tensor] = [input_ids[:, t] for t in range(T_prompt)]

        for _step in range(stop_len - T_prompt):
            # Build current prefix: (B, T_cur)
            prefix = lucid.stack(out_tokens, dim=1)
            outputs = cast(nn.Module, self)(prefix)
            logits = cast(
                Tensor, outputs.logits if hasattr(outputs, "logits") else outputs
            )
            next_logits = logits[:, -1, :]  # (B, vocab)

            # ── repetition penalty ────────────────────────────────────────
            if repetition_penalty != 1.0:
                next_logits = _apply_repetition_penalty(
                    next_logits, prefix, repetition_penalty
                )

            # ── greedy fast path ──────────────────────────────────────────
            if not do_sample:
                next_tok = lucid.argmax(next_logits, dim=-1)  # (B,)
            else:
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                if top_k is not None:
                    next_logits = _top_k_filter(next_logits, top_k)
                if top_p is not None:
                    next_logits = _top_p_filter(next_logits, top_p)
                probs = F.softmax(next_logits, dim=-1)  # (B, vocab)
                next_tok = _multinomial_one(probs, device=dev)

            # ── enforce padding for already-finished rows ────────────────
            next_list: list[int] = [int(next_tok[b].item()) for b in range(B)]
            for b in range(B):
                if finished[b]:
                    next_list[b] = pad_token_id
                elif eos_token_id is not None and next_list[b] == eos_token_id:
                    finished[b] = True
            out_tokens.append(lucid.tensor(next_list, device=dev).long())

            if all(finished):
                break

        return lucid.stack(out_tokens, dim=1).long()


# ─────────────────────────────────────────────────────────────────────────────
# Sampling primitives (module-private)
# ─────────────────────────────────────────────────────────────────────────────


def _apply_repetition_penalty(logits: Tensor, prefix: Tensor, penalty: float) -> Tensor:
    """Multiply logits of tokens already in ``prefix`` by ``1 / penalty``
    (or ``penalty`` if the logit is negative — HuggingFace convention).

    Args:
        logits: ``(B, vocab)`` next-token logits.
        prefix: ``(B, T)`` int tokens generated so far.
        penalty: Strictly positive float; ``> 1`` discourages repetition.
    """
    B = int(logits.shape[0])
    vocab = int(logits.shape[1])
    # Cheap row-by-row materialisation; vocab is large but T is bounded by
    # max_length so this loop is small.
    rows: list[list[float]] = [
        [float(logits[b, v].item()) for v in range(vocab)] for b in range(B)
    ]
    T = int(prefix.shape[1])
    for b in range(B):
        seen: set[int] = {int(prefix[b, t].item()) for t in range(T)}
        for tok in seen:
            if 0 <= tok < vocab:
                v = rows[b][tok]
                rows[b][tok] = v / penalty if v > 0 else v * penalty
    return lucid.tensor(rows, device=logits.device.type)


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Set every logit outside the top-K to ``-inf`` per row."""
    B = int(logits.shape[0])
    vocab = int(logits.shape[1])
    if k >= vocab:
        return logits
    NEG_INF = -1e9
    out_rows: list[list[float]] = []
    for b in range(B):
        row = [float(logits[b, v].item()) for v in range(vocab)]
        # k-th largest — partial sort via Python's heapq is fine for typical
        # vocab sizes (~50k) since this only runs once per generation step.
        threshold = sorted(row, reverse=True)[k - 1]
        out_rows.append([v if v >= threshold else NEG_INF for v in row])
    return lucid.tensor(out_rows, device=logits.device.type)


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep smallest set whose cumulative softmax ≥ p."""
    B = int(logits.shape[0])
    vocab = int(logits.shape[1])
    NEG_INF = -1e9
    out_rows: list[list[float]] = []
    for b in range(B):
        row = [float(logits[b, v].item()) for v in range(vocab)]
        # softmax on this row
        max_v = max(row)
        exps = [pow(2.71828182845904523536, v - max_v) for v in row]
        Z = sum(exps)
        probs = [e / Z for e in exps]
        # sort indices by descending prob
        order = sorted(range(vocab), key=lambda i: probs[i], reverse=True)
        cum = 0.0
        keep_mask = [False] * vocab
        for idx in order:
            cum += probs[idx]
            keep_mask[idx] = True
            if cum >= p:
                break
        out_rows.append([row[i] if keep_mask[i] else NEG_INF for i in range(vocab)])
    return lucid.tensor(out_rows, device=logits.device.type)


def _multinomial_one(probs: Tensor, *, device: str) -> Tensor:
    """Sample one token per row from ``(B, vocab)`` probability tensor.

    Lucid does not (yet) ship ``lucid.multinomial``, so we materialise the
    row, run the standard inverse-CDF sampling in Python on a single
    ``lucid.rand`` draw per row, and return ``(B,)`` int tokens.
    """
    B = int(probs.shape[0])
    vocab = int(probs.shape[1])
    u = lucid.rand((B,), device=device)
    out: list[int] = []
    for b in range(B):
        target = float(u[b].item())
        cum = 0.0
        chosen = vocab - 1
        for v in range(vocab):
            cum += float(probs[b, v].item())
            if cum >= target:
                chosen = v
                break
        out.append(chosen)
    return lucid.tensor(out, device=device).long()


# ─────────────────────────────────────────────────────────────────────────────
# DiffusionMixin — image-generation sampling loop
# ─────────────────────────────────────────────────────────────────────────────


class DiffusionMixin:
    """Standard sampling loop for diffusion-family generative models.

    Concrete subclasses must:

      * Inherit from :class:`lucid.models.PretrainedModel`.
      * Have a ``forward(sample, timestep, ...) -> DiffusionModelOutput`` that
        returns the network's prediction at the supplied timestep.
      * Expose a ``config`` attribute carrying ``sample_size`` / ``in_channels``
        (the fields :class:`DiffusionModelConfig` defines).

    The mixin then provides :meth:`generate` — a vanilla reverse-process loop
    parametrised by an externally-supplied :class:`Scheduler`.  Sampler
    choice (DDPM, DDIM, …) is therefore a runtime knob, not baked into the
    model class.
    """

    @lucid.no_grad()
    def generate(
        self,
        scheduler: Scheduler,
        *,
        n_samples: int = 1,
        num_inference_steps: int | None = None,
        generator_shape: tuple[int, ...] | None = None,
        return_intermediates: bool = False,
        device: str = "cpu",
    ) -> GenerationOutput:
        """Sample ``n_samples`` images via the reverse diffusion process.

        Args:
            scheduler:           Already-constructed :class:`Scheduler`.
            n_samples:           Batch size of the output.
            num_inference_steps: If ``None``, uses ``scheduler.timesteps`` as
                                 already set; otherwise calls
                                 ``scheduler.set_timesteps`` first.
            generator_shape:     Override the per-sample tensor shape (defaults
                                 to ``(in_channels, H, W)`` from ``self.config``).
            return_intermediates: If True, also returns the per-step samples.
            device:              Where to allocate the noise tensor.

        Returns:
            :class:`GenerationOutput` with the final ``(n_samples, C, H, W)``
            samples (and optional list of intermediates).
        """
        from lucid.models._output import GenerationOutput

        if num_inference_steps is not None:
            scheduler.set_timesteps(num_inference_steps, device=device)

        # Resolve image shape from the config when caller doesn't override.
        if generator_shape is None:
            cfg = getattr(self, "config", None)
            if cfg is None:
                raise RuntimeError(
                    "DiffusionMixin.generate needs either generator_shape or a "
                    "model.config with sample_size / in_channels fields."
                )
            sample_size = getattr(cfg, "sample_size", None)
            in_channels = getattr(cfg, "in_channels", None)
            if sample_size is None or in_channels is None:
                raise RuntimeError(
                    "model.config must provide sample_size and in_channels"
                )
            if isinstance(sample_size, tuple):
                H, W = int(sample_size[0]), int(sample_size[1])
            else:
                H = W = int(sample_size)
            generator_shape = (int(in_channels), H, W)

        full_shape = (n_samples, *generator_shape)
        sample = lucid.randn(full_shape, device=device)

        intermediates: list[Tensor] = []
        ts = scheduler.timesteps
        T_inf = int(ts.shape[0])
        for i in range(T_inf):
            t_int = int(ts[i].item())
            t_batch = lucid.tensor([t_int] * n_samples, device=device).long()
            output = cast(nn.Module, self)(sample, t_batch)
            model_pred = output.sample if hasattr(output, "sample") else output
            sample = scheduler.step(cast(Tensor, model_pred), t_int, sample)
            if return_intermediates:
                intermediates.append(sample)

        return GenerationOutput(
            samples=sample,
            intermediates=tuple(intermediates) if return_intermediates else None,
        )


# Re-import for the inner ``Scheduler`` forward reference above — the
# scheduler module imports lucid.nn.functional which transitively pulls in
# this mixin module, so we must keep the import lazy (inside generate).
