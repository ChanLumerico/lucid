#!/usr/bin/env python3
"""tools/check_pretrained_parity.py — the published checkpoint, run against
the implementation it was converted from.

The zoo's checks stack up to here and stop.  ``check_weight_fit`` asks
whether a checkpoint *loads*; the model parity suite asks whether the
architecture matches a reference, but transfers **random** weights to do
it.  Neither looks at the numbers actually published: a conversion that
transposed a kernel, mapped the wrong source, or dropped half a tensor
produces a file that loads into the right shapes and answers wrongly.

This loads the real checkpoint on both sides and compares outputs.  No
labels and no dataset are involved, so it runs anywhere the references
install — it is not an accuracy benchmark and does not pretend to be
one.  It catches conversion, not training.

**The reference is read, not guessed.**  Each entry's ``meta["source"]``
already records where its weights came from, and the two families in
this zoo do not share one: ``resnet_18`` is a reference-vision
checkpoint while ``sk_resnet_18`` is a timm one.  Comparing a model
against the wrong source shows every weight differing and looks exactly
like a defect — which is how this tool's first run was misread.

Text and CLIP checkpoints come from transformers.  Their outputs are
compared field by field — hidden states, logits, span logits, image and
text embeddings — relative to the reference's own scale, because a
language model's logits run to the hundreds and an absolute bound that
suits a classifier would fail them on float32 rounding.  The first of
these compared by hand, ``bert_base``, was 4.14 apart: the embedding
lookup zeroed the trained ``[PAD]`` row.  Models over ``_MAX_PARAMS`` are
reported rather than loaded: both copies at once do not fit a 16 GB
MacBook or a hosted runner.

Only sources whose reference package is installed can be checked; the
rest are reported as unreachable rather than skipped silently.

Run::

    python -m tools.check_pretrained_parity --limit 5
    python -m tools.check_pretrained_parity --model resnet_18_cls
    python -m tools.check_pretrained_parity --source timm
    python -m tools.check_pretrained_parity --source transformers
    python -m tools.check_pretrained_parity --list

Exit codes
----------
0 — every checkpoint that could be compared agrees.
1 — at least one disagrees beyond tolerance.
2 — nothing could be compared (no reference installed).
"""

import argparse
import importlib
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import lucid
import lucid.models  # noqa: F401 — populates the registry
from lucid.models import create_model
from lucid.models._registry import _REGISTRY
from lucid.weights._registry import _WEIGHTS_BY_MODEL
from lucid.test._fixtures.ref_framework import (
    ref_module,
    ref_vision_module,
    zoo_module,
)

#: Outputs agreeing to this are the same computation in a different order.
_ATOL = 1e-4

#: The same, relative to the reference's largest magnitude — for outputs
#: whose scale is not a classifier's.
_RTOL = 1e-4

#: Below this the comparison is against noise rather than a model.
_MIN_SCALE = 1e-3

#: Larger than this is reported, not loaded.  Two float32 copies of a
#: 774M-parameter model are over 6 GB before any activation.
_MAX_PARAMS = 500_000_000

#: The transformers class that reproduces each Lucid class, and the output
#: fields the two share.  A Lucid class not listed here is reported as
#: unsupported rather than compared against the wrong head.
_HF_HEADS: dict[str, tuple[str, tuple[str, ...]]] = {
    "BERTModel": ("AutoModel", ("last_hidden_state",)),
    "BERTForMaskedLM": ("AutoModelForMaskedLM", ("logits",)),
    "BERTForQuestionAnswering": (
        "AutoModelForQuestionAnswering",
        ("start_logits", "end_logits"),
    ),
    "BERTForTokenClassification": ("AutoModelForTokenClassification", ("logits",)),
    "GPTModel": ("AutoModel", ("last_hidden_state",)),
    "GPTLMHeadModel": ("AutoModelForCausalLM", ("logits",)),
    "GPT2Model": ("AutoModel", ("last_hidden_state",)),
    "GPT2LMHeadModel": ("AutoModelForCausalLM", ("logits",)),
    "RoFormerModel": ("AutoModel", ("last_hidden_state",)),
    "RoFormerForMaskedLM": ("AutoModelForMaskedLM", ("logits",)),
    "CvTForImageClassification": ("AutoModelForImageClassification", ("logits",)),
    "CLIPModel": ("CLIPModel", ("image_embeds", "text_embeds")),
}

#: CLIP's entries record only that OpenAI's weights were re-hosted by
#: transformers, not where; these are the repositories they came from.
_CLIP_REPOS: dict[str, tuple[str, int]] = {
    "clip_vit_base_16": ("openai/clip-vit-base-patch16", 224),
    "clip_vit_base_32": ("openai/clip-vit-base-patch32", 224),
    "clip_vit_large_14": ("openai/clip-vit-large-patch14", 224),
    "clip_vit_large_14_336": ("openai/clip-vit-large-patch14-336", 336),
}


def _reference_for(source: str) -> tuple[str, str] | None:
    """Split a source string into (package, identifier), or None.

    ``reference_vision/ResNet18_Weights.IMAGENET1K_V1`` names a weights
    enum in the reference vision package; ``timm/resnet18.a1_in1k`` names
    a model in the zoo oracle; ``transformers/google-bert/bert-base-uncased``
    names a repository; CLIP's entries say only that transformers
    re-hosted them.  Anything else — diffusers, darknet, a research repo —
    needs a loader this does not have.
    """
    if source.startswith("reference_vision/"):
        return ("vision", source.split("/", 1)[1])
    if source.startswith("timm/"):
        return ("timm", source.split("/", 1)[1])
    if source.startswith("transformers/"):
        return ("transformers", source.split("/", 1)[1])
    if source.startswith("openai/CLIPModel"):
        return ("clip", "")
    return None


def _transformers_module() -> object | None:
    try:
        return importlib.import_module("transformers")
    except ImportError:
        return None


def _build_reference(kind: str, identifier: str) -> object:
    if kind == "timm":
        timm = zoo_module()
        assert timm is not None
        # The whole identifier, tag included. ``resnext101_32x4d`` alone
        # loads timm's default tag, ``fb_ssl_yfcc100m_ft_in1k``, while this
        # zoo's checkpoint came from ``gluon_in1k`` — dropping the tag
        # compared two different checkpoints and reported the gap (5.75,
        # top-1 differing) as a conversion defect. seresnet50 the same:
        # ``a1_in1k`` by default, ``ra2_in1k`` here.
        return timm.create_model(identifier, pretrained=True).eval()

    vision = ref_vision_module()
    assert vision is not None
    enum_name, tag = identifier.split(".", 1)
    # ``ResNet18_Weights`` → ``resnet18``: the enum is named for the model
    # it belongs to, which is the only link between the two.  Keep the
    # underscores — ``ConvNeXt_Base_Weights`` is ``convnext_base`` and
    # ``VGG11_BN_Weights`` is ``vgg11_bn``; stripping them worked for
    # resnet and densenet by luck and broke four convnext entries into
    # what read like conversion defects.
    # Classifiers live on ``models``; segmenters and detectors live one
    # level down, and the source string does not say which. Look rather
    # than assume — the flat lookup reported six of them as conversion
    # defects when the enum was simply somewhere else.
    namespaces = [vision.models]
    for extra in ("segmentation", "detection"):
        try:
            namespaces.append(
                importlib.import_module(f"{vision.__name__}.models.{extra}")
            )
        except ImportError:  # pragma: no cover - depends on the install
            continue

    factory_name = enum_name.removesuffix("_Weights").lower()
    for namespace in namespaces:
        weights_enum = getattr(namespace, enum_name, None)
        factory = getattr(namespace, factory_name, None)
        if weights_enum is not None and factory is not None:
            return factory(weights=getattr(weights_enum, tag)).eval()

    raise AttributeError(
        f"the reference vision package has no {enum_name} paired with "
        f"{factory_name!r} in models, models.segmentation or "
        f"models.detection — the rule this derives one name from the "
        f"other with does not hold here"
    )


def _compare(
    model_name: str, source: str, shape: tuple[int, ...], params: int | None
) -> dict[str, object]:
    resolved = _reference_for(source)
    if resolved is None:
        return {"unreachable": f"no loader for {source.split('/')[0]!r}"}
    kind, identifier = resolved
    if kind in ("transformers", "clip"):
        return _compare_transformers(model_name, kind, identifier, params)
    if (kind == "timm" and zoo_module() is None) or (
        kind == "vision" and ref_vision_module() is None
    ):
        return {"unreachable": f"the {kind} reference is not installed"}

    ref = ref_module()
    if ref is None:
        return {"unreachable": "the reference framework is not installed"}

    try:
        ours = create_model(model_name, pretrained=True).eval()
        theirs = _build_reference(kind, identifier)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    try:
        with ref.no_grad():
            reference_out = theirs(ref.from_numpy(x.copy()))
        answer = ours(lucid.from_numpy(x.copy()))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    # Segmenters and detectors answer with a mapping, and the two sides
    # do not agree on its keys — a detector's output is a list of boxes
    # whose length depends on what it found, which is not a thing to
    # compare elementwise against another implementation's list. Say so
    # rather than reaching for a ``.numpy()`` that is not there.
    if not hasattr(reference_out, "numpy"):
        return {
            "unsupported": (
                f"the reference answers with "
                f"{type(reference_out).__name__}, not a tensor"
            )
        }

    wanted = reference_out.numpy()
    got = answer.logits.numpy() if hasattr(answer, "logits") else answer.numpy()

    scale = float(np.abs(wanted).max())
    if scale < _MIN_SCALE:
        return {"degenerate": f"the reference answers with |logits| up to {scale:.2g}"}

    return {
        "max_diff": float(np.abs(got - wanted).max()),
        "scale": scale,
        "top1_agrees": int(got.argmax()) == int(wanted.argmax()),
    }


def _compare_transformers(
    model_name: str, kind: str, identifier: str, params: int | None
) -> dict[str, object]:
    """One checkpoint against the transformers model it was converted from.

    Token ids are drawn at random below the vocabulary size, pixels from a
    unit normal; no tokenizer is involved, since the question is whether
    the same ids reach the same numbers, not what the text means.
    """
    transformers = _transformers_module()
    ref = ref_module()
    if transformers is None or ref is None:
        return {
            "unreachable": "transformers or the reference framework is not installed"
        }

    cls_name = _REGISTRY[model_name].model_class.__name__
    head = _HF_HEADS.get(cls_name)
    if head is None:
        return {"unsupported": f"no transformers head mapped for {cls_name}"}
    auto_name, fields = head
    repo, side = identifier, 224
    if kind == "clip":
        if model_name not in _CLIP_REPOS:
            return {"unsupported": f"no repository recorded for {model_name}"}
        repo, side = _CLIP_REPOS[model_name]
    if params is not None and params > _MAX_PARAMS:
        return {
            "unreachable": (
                f"{params / 1e6:.0f}M parameters — too large to hold both "
                f"copies here (limit {_MAX_PARAMS / 1e6:.0f}M)"
            )
        }

    try:
        ours = create_model(model_name, pretrained=True).eval()
        theirs = getattr(transformers, auto_name).from_pretrained(repo).eval()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    rng = np.random.default_rng(0)
    ours_args: list[object] = []
    theirs_kwargs: dict[str, object] = {}
    if cls_name in ("CvTForImageClassification", "CLIPModel"):
        pixels = rng.standard_normal((1, 3, side, side)).astype(np.float32)
        ours_args.append(lucid.from_numpy(pixels.copy()))
        theirs_kwargs["pixel_values"] = ref.from_numpy(pixels.copy())
    if cls_name != "CvTForImageClassification":
        vocab = int(ours.config.vocab_size)
        # CLIP's text tower takes exactly its context length and pools at
        # the [EOS] token, found as the largest id. Ending every sequence
        # on the top id pins that position on both sides, whichever way
        # the reference locates it.
        length = int(getattr(ours.config, "context_length", 16))
        ids = rng.integers(1, vocab - 1, size=(1, length)).astype(np.int64)
        if kind == "clip":
            ids[:, -1] = vocab - 1
        ours_args.append(lucid.from_numpy(ids.copy()).long())
        theirs_kwargs["input_ids"] = ref.from_numpy(ids.copy())

    try:
        with ref.no_grad():
            reference_out = theirs(**theirs_kwargs)
        answer = ours(*ours_args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    worst, scale, agrees = 0.0, 0.0, True
    for field in fields:
        wanted = getattr(reference_out, field).numpy()
        got = getattr(answer, field).numpy()
        if got.shape != wanted.shape:
            return {"error": f"{field}: shape {got.shape} against {wanted.shape}"}
        field_scale = float(np.abs(wanted).max())
        if field_scale < _MIN_SCALE:
            return {"degenerate": f"{field} reaches only {field_scale:.2g}"}
        worst = max(worst, float(np.abs(got - wanted).max()) / field_scale)
        scale = max(scale, field_scale)
        if "logits" in field:
            agrees = agrees and bool((got.argmax(-1) == wanted.argmax(-1)).all())

    return {
        "max_diff": worst,
        "scale": scale,
        "top1_agrees": agrees,
        "tolerance": _RTOL,
        "relative": True,
    }


def _params_of(name: str, member: object) -> int | None:
    meta = getattr(getattr(member, "value", None), "meta", {}) or {}
    count = meta.get("num_params") or getattr(_REGISTRY.get(name), "params", None)
    return int(count) if count else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", help="only this factory")
    parser.add_argument("--source", help="only sources starting with this")
    parser.add_argument("--limit", type=int, help="stop after this many")
    parser.add_argument("--list", action="store_true", help="print what is checkable")
    args = parser.parse_args()

    targets: list[tuple[str, str, int | None]] = []
    for name, enum in sorted(_WEIGHTS_BY_MODEL.items()):
        for tag, member in enum.__members__.items():
            if tag == "DEFAULT":
                continue
            source = str(member.value.meta.get("source", ""))
            if args.model and name != args.model:
                continue
            if args.source and not source.startswith(args.source):
                continue
            if _reference_for(source) is not None:
                targets.append((name, source, _params_of(name, member)))
            break  # one tag per factory is enough to catch a bad conversion

    if args.list:
        for name, source, _params in targets:
            print(f"{name}: {source}")
        print(f"\n{len(targets)} checkable")
        return 0
    if not targets:
        print("nothing selected — no entry matched", file=sys.stderr)
        return 1

    if args.limit:
        targets = targets[: args.limit]

    print(f"comparing {len(targets)} published checkpoints against their sources")
    bad: list[str] = []
    unreachable = 0
    checked = 0

    for index, (name, source, params) in enumerate(targets, 1):
        report = _compare(name, source, (1, 3, 224, 224), params)
        if "unreachable" in report:
            unreachable += 1
            print(f"  [{index}/{len(targets)}] {name:26s} — {report['unreachable']}")
            continue
        if "unsupported" in report:
            unreachable += 1
            print(f"  [{index}/{len(targets)}] {name:26s} — {report['unsupported']}")
            continue
        if "degenerate" in report:
            print(f"  [{index}/{len(targets)}] {name:26s} — {report['degenerate']}")
            continue
        if "error" in report:
            bad.append(f"  {name}: {report['error']}")
            print(f"  [{index}/{len(targets)}] {name:26s} ERROR", flush=True)
            continue
        checked += 1
        diff = float(report["max_diff"])  # type: ignore[arg-type]
        agrees = bool(report["top1_agrees"])
        tolerance = float(report.get("tolerance", _ATOL))  # type: ignore[arg-type]
        label = "rel|Δ|" if report.get("relative") else "max|Δ|"
        ok = diff <= tolerance and agrees
        print(
            f"  [{index}/{len(targets)}] {name:26s} {label}={diff:.2e} "
            f"top1={'=' if agrees else '≠'} {'ok' if ok else 'MISMATCH'}",
            flush=True,
        )
        if not ok:
            bad.append(
                f"  {name}: {label}={diff:.3e} against {source}"
                + ("" if agrees else " — and the top-1 class differs")
            )

    print(f"\n{checked} compared, {unreachable} unreachable")
    if bad:
        print(f"\n{len(bad)} problem(s):", file=sys.stderr)
        for line in bad:
            print(line, file=sys.stderr)
        print(
            "\nA published checkpoint that does not reproduce its source is a "
            "conversion defect: it loads, it runs, and it answers with "
            "something else.",
            file=sys.stderr,
        )
        return 1
    if checked == 0:
        print("no reference installed — nothing was verified", file=sys.stderr)
        return 2
    print("[check_pretrained_parity] OK — every checkpoint reproduces its source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
