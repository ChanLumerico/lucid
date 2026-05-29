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
    from lucid.models.generative._schedulers import DiffusionScheduler


@dataclass(frozen=True)
class FeatureInfo:
    r"""Specification of one stage emitted by a feature-extracting backbone.

    Used by detection / segmentation heads that consume a *feature
    pyramid*: each entry describes the channel count and spatial
    down-sampling factor of one backbone stage relative to the input.

    Attributes
    ----------
    stage : int
        Zero-based stage index, in the order the backbone emits features
        (typically deepening from stage 0 toward higher receptive field).
    num_channels : int
        Channel count of the feature map at this stage.
    reduction : int
        Spatial down-sampling factor vs the network input — e.g. ``4``,
        ``8``, ``16``, ``32`` for the canonical CNN stage taps.

    Notes
    -----
    The layout is timm-compatible: detection heads that consume timm
    backbones drop in unchanged.  See :class:`BackboneMixin` for the
    protocol that produces a ``list[FeatureInfo]``.

    Examples
    --------
    >>> info = FeatureInfo(stage=3, num_channels=512, reduction=32)
    >>> info.reduction
    32
    """

    stage: int
    num_channels: int
    reduction: int  # spatial down-sampling factor vs the input


class BackboneMixin(ABC):
    r"""Marker mixin for models usable as feature extractors.

    Detection and segmentation heads consume backbones through this small
    protocol: a ``forward_features(x)`` that emits the deepest stage's
    feature map, and a ``feature_info`` property enumerating every
    emitted stage's spec.  Implementing classes get a uniform interface
    that head builders (FPN, BiFPN, …) can target.

    Notes
    -----
    The mixin carries no state — it only declares two abstract slots.
    Subclasses MUST override:

    * :meth:`forward_features` — returns the final stage's feature map
      (and may stash intermediate stages on ``self`` for downstream
      use).
    * :attr:`feature_info` — returns a ``list[FeatureInfo]`` with one
      entry per stage the backbone exposes.

    Examples
    --------
    >>> class MyBackbone(PretrainedModel, BackboneMixin):
    ...     def forward_features(self, x):
    ...         return self.stages(x)
    ...     @property
    ...     def feature_info(self):
    ...         return [FeatureInfo(0, 64, 4), FeatureInfo(1, 128, 8)]
    """

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor:
        r"""Run the backbone's feature-extraction pass.

        Parameters
        ----------
        x : Tensor
            Input image batch, shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Feature map at the deepest stage.  Concrete subclasses may
            additionally stash earlier stages on the instance so
            multi-stage consumers can fetch them.
        """
        ...

    @property
    @abstractmethod
    def feature_info(self) -> list[FeatureInfo]:
        r"""Return the per-stage feature pyramid specification."""
        ...


class ClassificationHeadMixin:
    r"""Standard Linear classification head + transfer-learning helper.

    Adds two methods to any classifier subclass: :meth:`_build_classifier`
    constructs a ``Linear`` (optionally wrapped in a ``Dropout``)
    according to the config, and :meth:`reset_classifier` swaps in a
    freshly-initialised Linear with a new ``num_classes`` for transfer
    learning.

    Notes
    -----
    The contract: subclasses must call ``self._build_classifier(...)``
    inside their ``__init__`` *after* ``super().__init__(config)`` so the
    backbone is in place before the head is attached.

    The host class exposes ``self.classifier`` as either a plain
    ``nn.Linear`` (no dropout configured) or an ``nn.Sequential`` of
    ``[Dropout, Linear]``; :meth:`reset_classifier` handles both shapes.

    Examples
    --------
    >>> class MyClassifier(PretrainedModel, ClassificationHeadMixin):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         self.backbone = MyBackbone(config)
    ...         self._build_classifier(config.hidden_size, config.num_classes,
    ...                                dropout=config.dropout)
    >>> model = MyClassifier(cfg)
    >>> model.reset_classifier(num_classes=10)
    """

    classifier: nn.Module

    def _build_classifier(
        self,
        in_features: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        r"""Install the classification head on ``self.classifier``.

        Parameters
        ----------
        in_features : int
            Channel count entering the head (backbone pooled-feature
            dimension).
        num_classes : int
            Output class count.
        dropout : float, optional, keyword-only, default=0.0
            Drop probability.  When positive, the head becomes
            ``Sequential(Dropout(p), Linear(in, out))``; when zero, just a
            bare ``Linear``.

        Notes
        -----
        Called from the subclass ``__init__`` after the backbone is
        constructed.  Idempotent: calling twice replaces the previous
        head.
        """
        if dropout > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        r"""Replace the final ``Linear`` with a freshly initialised one.

        Parameters
        ----------
        num_classes : int
            New output dimensionality.  ``in_features`` is preserved from
            the current head.

        Raises
        ------
        RuntimeError
            If the head is a ``Sequential`` containing no ``Linear``.
        NotImplementedError
            If the head is some other custom module type.

        Notes
        -----
        Used during transfer learning — typically called immediately
        after :meth:`from_pretrained` to adapt a 1000-class ImageNet head
        to a downstream task with a different label set.

        Examples
        --------
        >>> model = AutoModelForImageClassification.from_pretrained("resnet_50")
        >>> model.reset_classifier(num_classes=10)
        >>> model(lucid.randn(1, 3, 224, 224)).logits.shape
        (1, 10)
        """
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
    r"""Per-token cross-entropy loss helper for masked-LM / token-class heads.

    Encoder text families (BERT, RoFormer, …) all reduce a ``(B, T, V)``
    logit tensor against ``(B, T)`` labels with the same recipe: flatten
    the sequence axis into the batch axis, then run
    :func:`lucid.nn.functional.cross_entropy` with ``ignore_index=-100``.
    Centralising the call here removes a 3-line stanza repeated across
    every ``ForMaskedLM`` / ``ForTokenClassification`` class.

    Notes
    -----
    The mixin is stateless and exposes a single ``@staticmethod`` —
    inheriting from it is purely a documentation / discoverability
    convenience; the same effect is obtained by calling
    ``MaskedLMMixin.compute_lm_loss(...)`` directly.

    Examples
    --------
    >>> class BERTForMaskedLM(PretrainedModel, MaskedLMMixin):
    ...     def forward(self, input_ids, labels=None):
    ...         logits = self.head(self.bert(input_ids))
    ...         loss = self.compute_lm_loss(logits, labels) if labels else None
    ...         return MaskedLMOutput(logits=logits, loss=loss)
    """

    @staticmethod
    def compute_lm_loss(
        logits: Tensor,
        labels: Tensor,
        *,
        ignore_index: int = -100,
    ) -> Tensor:
        r"""Compute per-token cross-entropy loss for an LM head.

        Parameters
        ----------
        logits : Tensor
            ``(B, T, V)`` per-token logits.
        labels : Tensor
            ``(B, T)`` int target token ids.  Entries equal to
            ``ignore_index`` contribute zero loss and zero gradient.
        ignore_index : int, optional, keyword-only, default=-100
            Token id to skip during reduction.  Convention from the wider
            ML ecosystem: ``-100`` marks non-masked positions in MLM and
            padding tokens in token classification.

        Returns
        -------
        Tensor
            Scalar loss tensor (mean over the contributing positions).

        Notes
        -----
        Implementation: flatten ``logits`` to ``(B*T, V)`` and ``labels``
        to ``(B*T,)``, cast labels to ``long``, then call
        :func:`lucid.nn.functional.cross_entropy`.  No probabilistic
        reweighting; the loss is a plain mean over non-ignored positions.

        Examples
        --------
        >>> logits = lucid.randn(2, 10, 32000)
        >>> labels = lucid.randint(0, 32000, (2, 10))
        >>> loss = MaskedLMMixin.compute_lm_loss(logits, labels)
        >>> loss.shape
        ()
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
    r"""Decoder-only autoregressive sampling for causal LM heads.

    Provides :meth:`generate` — greedy or stochastic next-token sampling
    with temperature, top-k, top-p (nucleus), and repetition-penalty
    knobs.  Per-sequence stopping at ``eos_token_id`` is honoured.

    Concrete subclasses must:

    * Inherit from :class:`lucid.models.PretrainedModel`.
    * Define ``forward(input_ids, ...)`` returning a
      :class:`lucid.models.CausalLMOutput` with ``logits`` of shape
      ``(B, T, vocab_size)``.
    * Expose a ``config`` attribute carrying ``vocab_size`` /
      ``pad_token_id`` / ``bos_token_id`` / ``eos_token_id`` (fields
      :class:`lucid.models.text.LanguageModelConfig` defines).

    Notes
    -----
    The first implementation does **not** use the model's KV cache —
    every step re-runs the full prefix.  This is correct but O(T²) in
    the prefix length.  Cache support is planned once GPT-2 lands and a
    concrete ``past_key_values`` shape is available to plumb through.

    Examples
    --------
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2_small")
    >>> prompt = lucid.tensor([[1, 2, 3]]).long()
    >>> out = model.generate(prompt, max_new_tokens=10, do_sample=True, top_p=0.9)
    >>> out.shape
    (1, 13)
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
        r"""Autoregressively extend ``input_ids`` until a stop condition.

        Parameters
        ----------
        input_ids : Tensor
            ``(B, T_prompt)`` int prompt tokens.
        max_length : int, optional, keyword-only, default=20
            Total sequence length cap (prompt + generated).  Ignored when
            ``max_new_tokens`` is supplied.
        max_new_tokens : int or None, optional, keyword-only
            Cap on the number of *new* tokens to generate, additive over
            the prompt length.  Takes precedence over ``max_length``.
        do_sample : bool, optional, keyword-only, default=False
            ``False`` → greedy argmax decoding; ``True`` → stochastic
            sampling honouring ``temperature`` / ``top_k`` / ``top_p``.
        temperature : float, optional, keyword-only, default=1.0
            Logit divisor before softmax.  ``< 1`` sharpens, ``> 1``
            flattens.  Ignored under greedy decoding.
        top_k : int or None, optional, keyword-only
            Keep only the K highest-probability tokens before sampling.
            ``None`` disables.
        top_p : float or None, optional, keyword-only
            Nucleus sampling — keep the smallest token set whose
            cumulative probability ≥ ``top_p``.  ``None`` disables.
        repetition_penalty : float, optional, keyword-only, default=1.0
            Multiply the logit of every previously generated token by
            ``1 / penalty`` (and by ``penalty`` if the logit is
            negative).  ``1.0`` → no effect; ``> 1`` discourages
            repetition.
        pad_token_id : int or None, optional, keyword-only
            Token id used to pad finished sequences after they emit
            ``eos_token_id``.  Defaults to ``config.pad_token_id``, then
            ``0`` if unset.
        eos_token_id : int or None, optional, keyword-only
            Stop generating per-sequence once this id is emitted.
            Defaults to ``config.eos_token_id``.

        Returns
        -------
        Tensor
            ``(B, T_final)`` int tensor where ``T_final`` ≤ ``max_length``
            (or ``T_prompt + max_new_tokens``).  Sequences that hit
            ``eos_token_id`` early are right-padded with ``pad_token_id``.

        Raises
        ------
        ValueError
            If ``input_ids`` is not 2-D.

        Notes
        -----
        Decorated with :func:`lucid.no_grad` — generation never builds an
        autograd graph.  Each step:

        1. Materialise the current prefix as ``(B, T_cur)``.
        2. Run a forward pass; take logits at the last position.
        3. Apply repetition penalty.
        4. Greedy argmax, or apply temperature / top-k / top-p and
           inverse-CDF sample one token per row.
        5. Replace finished rows' tokens with ``pad_token_id``.

        Loop breaks early when every row has emitted EOS.

        Examples
        --------
        >>> model = AutoModelForCausalLM.from_pretrained("gpt")
        >>> tokens = lucid.tensor([[1, 2, 3]]).long()
        >>> out = model.generate(tokens, max_new_tokens=5)
        >>> out.shape
        (1, 8)
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
    r"""Multiply logits of tokens already in ``prefix`` by ``1 / penalty``
    (or ``penalty`` if the logit is negative).

    Parameters
    ----------
    logits : Tensor
        ``(B, vocab)`` next-token logits.
    prefix : Tensor
        ``(B, T)`` int tokens generated so far.
    penalty : float
        Strictly positive float; ``> 1`` discourages repetition.

    Returns
    -------
    Tensor
        Adjusted ``(B, vocab)`` logits.

    Notes
    -----
    Convention popularised by the wider ML ecosystem: positive logits
    shrink toward 0, negative logits grow more negative — both directions
    push the affected token's probability down.
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
    r"""Set every logit outside the per-row top-K to ``-inf``.

    Parameters
    ----------
    logits : Tensor
        ``(B, vocab)`` logits.
    k : int
        Number of tokens to retain per row.  If ``k >= vocab``, ``logits``
        is returned unchanged.

    Returns
    -------
    Tensor
        Masked logits where non-top-K entries are replaced with a very
        large negative number (``-1e9``).
    """
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
    r"""Nucleus (top-p) filtering — keep the smallest token set with
    cumulative softmax probability ≥ ``p``.

    Parameters
    ----------
    logits : Tensor
        ``(B, vocab)`` logits.
    p : float
        Cumulative probability threshold in ``(0, 1]``.

    Returns
    -------
    Tensor
        Masked logits where tokens outside the nucleus are set to
        ``-1e9``.
    """
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
    r"""Draw exactly one token per row from a categorical distribution.

    Parameters
    ----------
    probs : Tensor
        ``(B, vocab)`` non-negative probability tensor — rows should sum
        to 1.
    device : str, keyword-only
        Device string to allocate the random draws on.

    Returns
    -------
    Tensor
        ``(B,)`` long tensor of sampled token ids.

    Notes
    -----
    Lucid does not (yet) ship ``lucid.multinomial`` so this routine uses
    a Python inverse-CDF loop on a single uniform draw per row.
    Acceptable because it only runs once per generation step.
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
    r"""Reverse-process sampling loop for diffusion-family generative models.

    Concrete subclasses must:

    * Inherit from :class:`lucid.models.PretrainedModel`.
    * Define ``forward(sample, timestep, ...) -> DiffusionModelOutput``
      that returns the network's prediction at the supplied timestep.
    * Expose a ``config`` attribute carrying ``sample_size`` /
      ``in_channels`` (fields :class:`DiffusionModelConfig` defines) — or
      explicitly pass ``generator_shape`` to :meth:`generate`.

    The mixin then provides :meth:`generate` — a vanilla reverse-process
    loop parametrised by an externally-supplied
    :class:`DiffusionScheduler`.  Sampler choice (DDPM, DDIM, …) is thus
    a runtime knob rather than baked into the model class.

    Notes
    -----
    The mixin is stateless.  All hyper-parameters (number of inference
    steps, image shape, batch size) flow through :meth:`generate`'s
    keyword arguments; the model class itself stays sampler-agnostic.

    Examples
    --------
    >>> from lucid.models import DDPMScheduler, AutoModelForImageGeneration
    >>> model = AutoModelForImageGeneration.from_pretrained("ddpm_cifar_gen")
    >>> scheduler = DDPMScheduler(num_train_timesteps=1000)
    >>> out = model.generate(scheduler, n_samples=4, num_inference_steps=50)
    >>> out.samples.shape
    (4, 3, 32, 32)
    """

    @lucid.no_grad()
    def generate(
        self,
        scheduler: DiffusionScheduler,
        *,
        n_samples: int = 1,
        num_inference_steps: int | None = None,
        generator_shape: tuple[int, ...] | None = None,
        return_intermediates: bool = False,
        device: str = "cpu",
    ) -> GenerationOutput:
        r"""Sample ``n_samples`` images via the reverse diffusion process.

        Parameters
        ----------
        scheduler : DiffusionScheduler
            Already-constructed scheduler (DDPM, DDIM, etc.).  Encapsulates
            the noise schedule and the per-step update rule.
        n_samples : int, optional, keyword-only, default=1
            Batch size of the generated output.
        num_inference_steps : int or None, optional, keyword-only
            If ``None``, the scheduler's ``timesteps`` are used as-is;
            otherwise ``scheduler.set_timesteps(num_inference_steps, ...)``
            is called first to (re)build the timestep schedule.
        generator_shape : tuple[int, ...] or None, optional, keyword-only
            Per-sample tensor shape, ``(C, H, W)``.  When omitted,
            defaults to ``(in_channels, H, W)`` derived from
            ``self.config``.
        return_intermediates : bool, optional, keyword-only, default=False
            If ``True``, every per-step sample is recorded and returned
            in :attr:`GenerationOutput.intermediates`.
        device : str, optional, keyword-only, default="cpu"
            Device to allocate the initial Gaussian noise tensor on.

        Returns
        -------
        GenerationOutput
            ``.samples`` is the final ``(n_samples, C, H, W)`` batch;
            ``.intermediates`` is populated when ``return_intermediates``
            is ``True``.

        Raises
        ------
        RuntimeError
            If ``generator_shape`` is ``None`` and the model lacks a
            ``config`` with ``sample_size`` / ``in_channels`` fields.

        Notes
        -----
        Decorated with :func:`lucid.no_grad` — sampling never builds an
        autograd graph.  Standard reverse-process loop:

        1. Draw :math:`x_T \sim \mathcal{N}(0, I)`.
        2. For each ``t`` in ``scheduler.timesteps`` (descending):
           run the model forward, then apply
           ``scheduler.step(model_pred, t, x_t)`` to obtain :math:`x_{t-1}`.

        Examples
        --------
        >>> model = AutoModelForImageGeneration.from_pretrained("ddpm_cifar_gen")
        >>> scheduler = DDPMScheduler(num_train_timesteps=1000)
        >>> out = model.generate(scheduler, n_samples=4, num_inference_steps=50,
        ...                      device="cpu")
        >>> out.samples.shape
        (4, 3, 32, 32)
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
