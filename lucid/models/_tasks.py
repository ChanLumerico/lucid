"""The eight tasks a model in this zoo can be built for.

Every ``<Family>For<Task>`` wrapper inherits one of these.  Before this
existed all seventy-four of them subclassed :class:`PretrainedModel`
directly, so "what task is this model for" was answerable only by reading
the class *name* — and three different answers were in circulation at
once: fifteen distinct ``For<Task>`` suffixes, thirteen ``task=`` tags in
the registry, and twelve ``AutoModelFor*`` shells, with nothing marking
which was authoritative.

The taxonomy is deliberately coarse.  A finer one grows a task every time
a family arrives with a slightly different head — the zoo had reached one
task used by a single family six times over, at which point the "task" was
really just a suffix on a class name.  Where a family's head is a
specialisation, it inherits the general task and keeps its specific name:
:class:`~lucid.models.CLIPForZeroShotImageClassification` is an
:class:`ImageClassificationModel`, and zero-shot is what the class is
called, not a task of its own.  Reasoning and the alternatives that were
rejected: ``obsidian/architecture/arch-models-task-taxonomy.md``.

**What a task base does and does not promise.**  It fixes the *structure*:
the config contract, the pretrained-checkpoint flow, and which
:mod:`~lucid.models._output` dataclass family the forward returns.  It
does **not** fix the forward signature, and callers must not assume two
models under one task are interchangeable.  The dividing line, measured
across the zoo, is whether the second positional parameter is required::

    ResNetForImageClassification.forward(x, labels=None)
    CLIPForZeroShotImageClassification.forward(pixel_values, prompt_ids)
    TransformerForSeq2SeqLM.forward(input_ids, decoder_input_ids, ...)

CLIP cannot classify without prompts, and making ``prompt_ids`` optional
to satisfy a shared signature would break the model to flatter the
taxonomy.  So the promise stops at structure, and
:meth:`~lucid.models.AutoModel.from_pretrained` hands back something you
still have to call correctly for its family.
"""

from typing import ClassVar

from lucid.models._base import PretrainedModel

__all__ = [
    "TaskModel",
    "ImageClassificationModel",
    "ImageGenerationModel",
    "ObjectDetectionModel",
    "SemanticSegmentationModel",
    "SequenceClassificationModel",
    "TokenClassificationModel",
    "LanguageModelingModel",
    "WorldModelingModel",
]


class TaskModel(PretrainedModel):
    """Common base of the eight task bases.

    Exists so ``isinstance(model, TaskModel)`` distinguishes a head-bearing
    wrapper from a bare backbone, which is otherwise only inferable from
    the class name.  Carries the canonical task tag that
    ``@register_model(task=...)`` and the ``AutoModelFor*`` shells agree
    on — one string, defined once, rather than the three parallel lists
    this replaces.

    Attributes
    ----------
    task : ClassVar[str]
        The registry tag for this task.  ``""`` on this class because it
        names no task; every concrete base overrides it.
    """

    task: ClassVar[str] = ""


class ImageClassificationModel(TaskModel):
    """A model that assigns labels to an image.

    Returns :class:`~lucid.models.ImageClassificationOutput`, except where
    the head produces a richer result — CLIP's zero-shot wrapper scores
    images against text prompts and returns its own output type.  That is
    the specialisation this base is deliberately coarse enough to hold.
    """

    task: ClassVar[str] = "image-classification"


class ImageGenerationModel(TaskModel):
    """A model that samples images.

    Spans the diffusion, score-based, flow and autoencoder families, whose
    sampling procedures have nothing in common beyond producing pixels —
    which is why the base fixes the output family and leaves ``generate``
    to the mixins in :mod:`lucid.models._mixins`.
    """

    task: ClassVar[str] = "image-generation"


class ObjectDetectionModel(TaskModel):
    """A model that localises and labels objects.

    Returns :class:`~lucid.models.ObjectDetectionOutput`, or
    :class:`~lucid.models.InstanceSegmentationOutput` where the head also
    predicts masks (Mask R-CNN) — the mask is an addition to detection
    rather than a different task.
    """

    task: ClassVar[str] = "object-detection"


class SemanticSegmentationModel(TaskModel):
    """A model that labels every pixel.

    Returns :class:`~lucid.models.SemanticSegmentationOutput`.
    """

    task: ClassVar[str] = "semantic-segmentation"


class SequenceClassificationModel(TaskModel):
    """A model that assigns labels to a whole sequence.

    Also the home of the sentence-pair and span heads — next-sentence
    prediction, multiple choice and extractive question answering all read
    a pooled representation and score it, and each was used by exactly one
    family when they were separate tasks.  Question answering returns
    :class:`~lucid.models.QuestionAnsweringOutput` rather than the
    sequence-classification one; the shape of the answer differs, the task
    it serves does not.
    """

    task: ClassVar[str] = "sequence-classification"


class TokenClassificationModel(TaskModel):
    """A model that labels every token.

    Returns :class:`~lucid.models.TokenClassificationOutput`.  Kept apart
    from sequence classification because the head shape genuinely differs
    — per-token logits against one pooled vector — rather than because the
    families differ.
    """

    task: ClassVar[str] = "token-classification"


class LanguageModelingModel(TaskModel):
    """A model that predicts tokens from context.

    Covers the masked, causal and sequence-to-sequence objectives.  They
    were three tasks and are one: each trains a language model and the
    difference is which positions are visible when predicting a token.
    The output types stay distinct
    (:class:`~lucid.models.MaskedLMOutput`,
    :class:`~lucid.models.CausalLMOutput`,
    :class:`~lucid.models.Seq2SeqLMOutput`), and the encoder-decoder
    variants need a second required argument, so this is the base where
    the "not interchangeable" warning in the module docstring bites
    hardest.
    """

    task: ClassVar[str] = "language-modeling"


class WorldModelingModel(TaskModel):
    """A model that learns an environment's latent dynamics.

    Returns the family's own output dataclass — these carry a recurrent
    state rather than logits, and no shared output type would describe
    them without inventing one no caller reads.
    """

    task: ClassVar[str] = "world-modeling"
