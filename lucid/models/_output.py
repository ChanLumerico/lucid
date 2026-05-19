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
    r"""Base for all model-forward output dataclasses.

    Provides a uniform protocol: outputs are simultaneously dataclass-like
    (named fields, dict-style lookup) and sequence-like (iteration / tuple
    indexing, with ``None`` fields silently elided).  Each concrete
    subclass adds a task-specific set of ``Tensor`` (or
    ``tuple[Tensor, ...]``) fields.

    Notes
    -----
    Subclasses must be decorated with ``@dataclass`` so
    :func:`dataclasses.fields` works.  Iteration order follows field
    declaration order, and ``None`` fields are skipped — this matches the
    behaviour expected by callers that destructure outputs with
    ``logits, loss = model(x)``.

    Direct instantiation of :class:`ModelOutput` itself is not useful (no
    fields) — always subclass and add ``@dataclass``.

    Examples
    --------
    >>> # ImageClassificationOutput is a typical subclass
    >>> out = ImageClassificationOutput(logits=lucid.randn(1, 10))
    >>> out["logits"].shape           # dict-style access
    (1, 10)
    >>> first, = out                  # tuple-style unpacking
    >>> first is out.logits
    True
    """

    def __iter__(self) -> Iterator[Tensor]:
        r"""Yield each non-``None`` field value in declaration order."""
        for f in fields(self):  # type: ignore[arg-type]
            value: Tensor | None = getattr(self, f.name)
            if value is not None:
                yield value

    def __getitem__(self, idx: int | str) -> Tensor:
        r"""Look up a field by name (``str``) or position (``int``).

        Parameters
        ----------
        idx : int or str
            Field name or positional index into the non-``None``
            iteration.

        Returns
        -------
        Tensor
            The requested field value.

        Raises
        ------
        KeyError
            If ``idx`` is a string naming a missing / ``None`` field.
        IndexError
            If ``idx`` is an integer outside the non-``None`` range.
        """
        if isinstance(idx, str):
            value2: Tensor | None = getattr(self, idx, None)
            if value2 is None:
                raise KeyError(idx)
            return value2
        return tuple(self)[idx]

    def __len__(self) -> int:
        r"""Return the number of non-``None`` fields."""
        count = 0
        for f in fields(self):  # type: ignore[arg-type]
            if getattr(self, f.name) is not None:
                count += 1
        return count

    def __contains__(self, key: object) -> bool:
        r"""Return whether ``key`` names a present (non-``None``) field."""
        if not isinstance(key, str):
            return False
        return getattr(self, key, None) is not None

    def to_tuple(self) -> tuple[Tensor, ...]:
        r"""Return a tuple of non-``None`` field values in declaration order.

        Examples
        --------
        >>> out = BaseModelOutput(last_hidden_state=h)
        >>> out.to_tuple()
        (h,)
        """
        return tuple(self)

    def keys(self) -> list[str]:
        r"""Names of fields whose value is not ``None``, in declaration order."""
        return [
            f.name
            for f in fields(self)  # type: ignore[arg-type]
            if getattr(self, f.name) is not None
        ]

    def values(self) -> list[Tensor]:
        r"""Values of non-``None`` fields, in declaration order."""
        return list(self)

    def items(self) -> list[tuple[str, Tensor]]:
        r"""``(name, value)`` pairs of non-``None`` fields."""
        return [(k, getattr(self, k)) for k in self.keys()]


@dataclass
class BaseModelOutput(ModelOutput):
    r"""Generic backbone forward output — sequence model variant.

    Returned by backbone classes that emit a single hidden-state stream
    (encoder-only and decoder-only language models, ViT-family backbones).

    Attributes
    ----------
    last_hidden_state : Tensor
        Final layer's output, shaped ``(B, T, H)`` for sequence models or
        ``(B, C, H, W)`` for spatial backbones.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states when the caller passed
        ``output_hidden_states=True`` (else ``None``).
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights when the caller passed
        ``output_attentions=True`` (else ``None``).

    Notes
    -----
    Produced by transformer-family backbones (BERT, RoFormer, ViT, …) and
    most other encoder-only architectures that don't need a pooled
    representation.

    Examples
    --------
    >>> model = AutoModel.from_pretrained("bert_base")
    >>> out = model(input_ids)
    >>> out.last_hidden_state.shape
    (1, 128, 768)
    """

    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
    r"""Backbone output + pooled representation (e.g. CLS).

    Attributes
    ----------
    last_hidden_state : Tensor
        Per-token / per-spatial-location output, shape ``(B, T, H)``.
    pooler_output : Tensor
        Aggregated representation — typically the ``[CLS]`` position
        passed through ``tanh(Linear(H, H))`` (BERT-style) or a learned
        attention pool.  Shape ``(B, H)``.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states; populated when
        ``output_hidden_states=True``.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights; populated when
        ``output_attentions=True``.

    Notes
    -----
    Used by encoder LMs that ship a sentence-level pooling head out of
    the box — BERT and friends.  Downstream classifiers attach to
    ``pooler_output`` rather than re-pooling the per-token sequence.

    Examples
    --------
    >>> model = AutoModel.from_pretrained("bert_base")
    >>> out = model(input_ids)
    >>> out.pooler_output.shape
    (1, 768)
    """

    last_hidden_state: Tensor
    pooler_output: Tensor
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class ImageClassificationOutput(ModelOutput):
    r"""Output of any ``{Family}ForImageClassification`` model.

    Attributes
    ----------
    logits : Tensor
        Pre-softmax class scores, shape ``(B, num_classes)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss when ``labels`` were supplied to
        ``forward``; otherwise ``None``.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer feature maps when requested via
        ``output_hidden_states=True``.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights for transformer-family classifiers.

    Notes
    -----
    Returned by every classifier registered under
    :class:`AutoModelForImageClassification` — CNN-family
    (ResNet, EfficientNet, …) and ViT-family backbones alike.

    Examples
    --------
    >>> model = AutoModelForImageClassification.from_pretrained("resnet_50")
    >>> out = model(lucid.randn(1, 3, 224, 224))
    >>> out.logits.shape
    (1, 1000)
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class ObjectDetectionOutput(ModelOutput):
    r"""Output of any ``{Family}ForObjectDetection`` model.

    Attributes
    ----------
    logits : Tensor
        Per-proposal / per-query class logits.  Shape depends on family:
        ``(B, num_queries, num_classes)`` for DETR, ``(N, num_classes)``
        for R-CNN-family per-RoI scoring.
    pred_boxes : Tensor
        Predicted bounding box coordinates.  Box format is family-specific
        (cxcywh-normalised for DETR, xyxy-pixel for R-CNN).
    loss : Tensor or None, optional
        Total detection loss (classification + box regression + matching
        cost where applicable) when targets were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Optional intermediate feature maps.
    proposals : tuple[Tensor, ...] or None, optional
        Per-image RoI proposals for two-stage detectors (R-CNN / Fast
        R-CNN / Faster R-CNN).  Lets downstream ``postprocess()`` run
        from the output alone, without re-running the proposal stage.

    Notes
    -----
    Returned by :class:`AutoModelForObjectDetection` registrations —
    R-CNN family, DETR, EfficientDet, YOLO v1–v4.  Anchor-based and
    set-prediction families share this contract.

    Examples
    --------
    >>> model = AutoModelForObjectDetection.from_pretrained("detr_resnet50")
    >>> out = model(lucid.randn(1, 3, 800, 800))
    >>> out.logits.shape, out.pred_boxes.shape
    ((1, 100, 91), (1, 100, 4))
    """

    logits: Tensor
    pred_boxes: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    # Two-stage detectors (R-CNN / Fast R-CNN / Faster R-CNN) emit detections
    # *per RoI*; ``proposals`` carries the per-image RoI list so downstream
    # ``postprocess()`` can be called against the output alone.
    proposals: tuple[Tensor, ...] | None = None


@dataclass
class InstanceSegmentationOutput(ModelOutput):
    r"""Output of any instance-segmentation model.

    Attributes
    ----------
    logits : Tensor
        Per-proposal class logits.
    pred_boxes : Tensor
        Predicted bounding boxes.
    pred_masks : Tensor
        Per-instance binary mask logits, typically shape
        ``(N, num_classes, mh, mw)`` for R-CNN-style or
        ``(B, num_queries, H, W)`` for transformer-based instance heads.
    loss : Tensor or None, optional
        Total loss (cls + box + mask) when targets were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Optional intermediate feature maps.

    Notes
    -----
    Returned today by Mask R-CNN; Mask2Former when configured for
    instance segmentation will also produce this shape.

    Examples
    --------
    >>> model = create_model("mask_rcnn")
    >>> out = model(lucid.randn(1, 3, 800, 800))
    >>> out.pred_masks.shape[-2:]
    (28, 28)
    """

    logits: Tensor
    pred_boxes: Tensor
    pred_masks: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass
class SemanticSegmentationOutput(ModelOutput):
    r"""Output of any ``{Family}ForSemanticSegmentation`` model.

    Attributes
    ----------
    logits : Tensor
        Per-pixel class logits, shape ``(B, num_classes, H, W)``.  Spatial
        resolution may match the input or be downsampled — call
        ``F.interpolate`` to upsample when needed.
    loss : Tensor or None, optional
        Scalar segmentation loss (cross-entropy / Dice / per-class) when
        targets were supplied.
    hidden_states : tuple[Tensor, ...] or None, optional
        Optional encoder / decoder feature maps.

    Notes
    -----
    Returned by FCN, U-Net, Attention U-Net, MaskFormer, and Mask2Former
    when used for semantic segmentation.

    Examples
    --------
    >>> model = AutoModelForSemanticSegmentation.from_pretrained("fcn_resnet50")
    >>> out = model(lucid.randn(1, 3, 512, 512))
    >>> out.logits.shape
    (1, 21, 512, 512)
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass
class CausalLMOutput(ModelOutput):
    r"""Output of any causal (left-to-right) language model head.

    Attributes
    ----------
    logits : Tensor
        Per-position vocabulary logits, shape ``(B, T, vocab_size)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss when ``labels`` were supplied.  Labels
        are typically shifted internally so ``loss`` reflects next-token
        prediction.
    past_key_values : tuple[tuple[Tensor, Tensor], ...] or None, optional
        Per-layer ``(key, value)`` cache for incremental decoding.  Each
        tensor shape ``(B, num_heads, T, head_dim)``.
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Produced by GPT-1, GPT-2 (``GPT2LMHeadModel``), and any future
    decoder-only LM head.  :class:`GenerationMixin` consumes the
    ``logits`` field; future cache-aware sampling will plumb through
    ``past_key_values``.

    Examples
    --------
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2_small")
    >>> out = model(input_ids)
    >>> out.logits.shape
    (1, 128, 50257)
    """

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class MaskedLMOutput(ModelOutput):
    r"""Output of any masked-LM head (BERT, RoFormer, …).

    Attributes
    ----------
    logits : Tensor
        Per-position vocabulary logits, shape ``(B, T, vocab_size)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss reduced over masked positions only
        (``ignore_index=-100`` for non-masked tokens).
    hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer hidden states.
    attentions : tuple[Tensor, ...] or None, optional
        Per-layer attention weights.

    Notes
    -----
    Returned by every ``{Family}ForMaskedLM`` registration —
    ``BertForMaskedLM``, ``RoFormerForMaskedLM``, and future encoder LM
    heads.  Loss is computed via :meth:`MaskedLMMixin.compute_lm_loss`.

    Examples
    --------
    >>> model = AutoModelForMaskedLM.from_pretrained("bert_base_mlm")
    >>> out = model(input_ids, labels=labels)
    >>> out.loss.item()
    2.34
    """

    logits: Tensor
    loss: Tensor | None = None
    hidden_states: tuple[Tensor, ...] | None = None
    attentions: tuple[Tensor, ...] | None = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    r"""Output of any encoder-decoder seq2seq language model.

    Attributes
    ----------
    logits : Tensor
        Decoder vocabulary logits, shape ``(B, T_dec, vocab_size)``.
    loss : Tensor or None, optional
        Scalar cross-entropy loss when ``labels`` were supplied (label
        smoothing / shift handled internally).
    past_key_values : tuple[tuple[Tensor, Tensor], ...] or None, optional
        Per-layer decoder ``(key, value)`` cache for autoregressive
        decoding.
    decoder_hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer decoder hidden states.
    decoder_attentions : tuple[Tensor, ...] or None, optional
        Per-layer decoder self-attention weights.
    encoder_last_hidden_state : Tensor or None, optional
        Final encoder output, cached so callers can run multiple decoder
        passes (e.g. beam search) without re-encoding.
    encoder_hidden_states : tuple[Tensor, ...] or None, optional
        Per-layer encoder hidden states.
    encoder_attentions : tuple[Tensor, ...] or None, optional
        Per-layer encoder self-attention weights.

    Notes
    -----
    Produced by ``TransformerForSeq2SeqLM`` today; T5, BART, and mBART
    are the natural future consumers.  Caching the encoder output is the
    standard pattern for batched generation.

    Examples
    --------
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("transformer_base_seq2seq")
    >>> out = model(input_ids=src, decoder_input_ids=tgt)
    >>> out.logits.shape
    (1, 16, 32000)
    """

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    decoder_hidden_states: tuple[Tensor, ...] | None = None
    decoder_attentions: tuple[Tensor, ...] | None = None
    encoder_last_hidden_state: Tensor | None = None
    encoder_hidden_states: tuple[Tensor, ...] | None = None
    encoder_attentions: tuple[Tensor, ...] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Generative-family outputs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiffusionModelOutput(ModelOutput):
    r"""Single-step output of a diffusion U-Net's forward pass.

    Attributes
    ----------
    sample : Tensor
        Network prediction at the supplied timestep.  The semantic
        interpretation depends on the parameterisation declared in
        :attr:`DiffusionModelConfig.prediction_type`:

        * ``"epsilon"`` — predicted noise :math:`\varepsilon`.
        * ``"sample"`` — predicted clean signal :math:`x_0`.
        * ``"v"`` — predicted velocity :math:`v = \alpha_t \varepsilon -
          \sigma_t x_0`.
    loss : Tensor or None, optional
        Scalar training loss against ground-truth noise (MSE for the
        ``"epsilon"`` parameterisation).

    Notes
    -----
    Returned by ``DDPMUNet.forward`` and other diffusion noise predictors.
    The sampling loop in :class:`DiffusionMixin` reads ``sample`` and
    passes it to the scheduler's ``step`` method.

    Examples
    --------
    >>> unet = DDPMUNet(cfg)
    >>> out = unet(x_noisy, t)
    >>> out.sample.shape == x_noisy.shape
    True
    """

    sample: Tensor
    loss: Tensor | None = None


@dataclass
class VAEOutput(ModelOutput):
    r"""End-to-end VAE forward output.

    Attributes
    ----------
    sample : Tensor
        Reconstructed image :math:`\hat{x}` shaped ``(B, C, H, W)``.
    latent : Tensor
        Sampled latent :math:`z = \mu + \sigma \cdot \varepsilon`, shape
        ``(B, latent_dim)``.
    mu : Tensor
        Encoder mean :math:`\mu(x)`.
    logvar : Tensor
        Encoder log-variance :math:`\log \sigma^2(x)`.
    loss : Tensor or None, optional
        Total ELBO loss (``recon_loss + β · kl_loss``) when targets were
        supplied to ``forward``.
    recon_loss : Tensor or None, optional
        Reconstruction term alone (typically BCE or MSE).
    kl_loss : Tensor or None, optional
        KL divergence term :math:`D_{\mathrm{KL}}(q(z|x) \| p(z))` alone.

    Notes
    -----
    Returned by :class:`VAEModel.forward`.  :class:`VAEForImageGeneration`
    wraps the decoder for unconditional sampling and returns
    :class:`GenerationOutput` from its ``generate`` method.

    Examples
    --------
    >>> model = create_model("vae")
    >>> out = model(images)
    >>> out.sample.shape, out.mu.shape
    ((4, 3, 64, 64), (4, 128))
    """

    sample: Tensor
    latent: Tensor
    mu: Tensor
    logvar: Tensor
    loss: Tensor | None = None
    recon_loss: Tensor | None = None
    kl_loss: Tensor | None = None


@dataclass
class GenerationOutput(ModelOutput):
    r"""Final result of a generative model's sampling loop.

    Attributes
    ----------
    samples : Tensor
        Final ``(n_samples, C, H, W)`` (image) or ``(n_samples, T)``
        (text) batch produced by the sampler.
    intermediates : tuple[Tensor, ...] or None, optional
        Per-step latents / samples — populated only when the caller
        passes ``return_intermediates=True`` to ``generate``.  Useful for
        animating trajectories or debugging schedulers.

    Notes
    -----
    Returned by :meth:`DiffusionMixin.generate` and
    :class:`VAEForImageGeneration.generate`.  Text generation through
    :meth:`GenerationMixin.generate` currently returns a bare
    :class:`Tensor` rather than this wrapper — that may change.

    Examples
    --------
    >>> model = AutoModelForImageGeneration.from_pretrained("ddpm_cifar_gen")
    >>> scheduler = DDPMScheduler(num_train_timesteps=1000)
    >>> out = model.generate(scheduler, n_samples=4, num_inference_steps=50)
    >>> out.samples.shape
    (4, 3, 32, 32)
    """

    samples: Tensor
    intermediates: tuple[Tensor, ...] | None = None
