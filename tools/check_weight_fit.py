#!/usr/bin/env python3
"""tools/check_weight_fit.py — assert every published checkpoint still fits
the factory that offers it.

A factory and the checkpoint it ships drift apart silently.  Nothing about
``num_encoder_layers=6`` looks wrong next to a config whose default is 0,
and the model builds, trains and exports either way — the only thing that
breaks is ``pretrained=True``, which no test exercised.  Three families
were shipping unloadable weights when this was written:

  * ``sk_resnet_18/34_cls`` — a paper floor (L=32) the checkpoints were
    trained without, so the two narrow stages wanted 32-wide attention
    against a 16-wide checkpoint.
  * ``maskformer_resnet50/101`` — a 6-layer transformer encoder the
    checkpoints have no weights for, contradicting the config's own
    comment.
  * ``resnest_200/269_cls`` — a dropout that moves the classifier to
    ``classifier.1``, against a checkpoint converted from a flat head.

**No weight data is downloaded.**  A safetensors file opens with a
length-prefixed JSON header naming every tensor and its shape, so two
HTTP range requests (a few KB) answer the question for a checkpoint of
any size.

**The factory is called, not imitated.**  An earlier version of this
check rebuilt the model itself and reported six false positives, because
factories do things a reimplementation does not: load into a submodule
(``model.dit``, ``model.diamond``), derive config from the entry
(per-game action counts), or switch config on whether overrides were
passed.  Here ``load_weight_entry`` is intercepted instead — the factory
runs its real path and hands over the very module the load targets, then
the download is skipped.

What a pass does *not* mean
---------------------------
Structure, not identity.  This reads the header and compares names and
shapes; it never hashes the file.  A checkpoint replaced by a different
revision of the same architecture passes here and is caught only by
``_verify_sha256`` at download time, which needs all the bytes.

There is no cheap stand-in for that: the Hub's ``ETag`` is not the
content digest — checked against four entries that download and verify
cleanly, and it differed from the declared SHA-256 for every one.

And a fit says the weights *load*, not that they are any good.  No
accuracy is verified anywhere in this repository; see
``obsidian/api/api-python-models.md`` on registration not being
validation.

Run::

    python -m tools.check_weight_fit                   # every checkpoint
    python -m tools.check_weight_fit --family sknet    # one family
    python -m tools.check_weight_fit --model resnet_18_cls
    python -m tools.check_weight_fit --list            # print, do not check

Exit codes
----------
0 — every checkpoint fits, or none was reachable to test.
1 — at least one checkpoint does not fit its factory.
2 — the network could not be reached for some checkpoint (reported
    separately from a mismatch, because "unknown" is not "broken").
"""

import argparse
import json
import struct
import sys
import urllib.error
import urllib.request
import warnings
from contextlib import contextmanager
from typing import Iterator

warnings.filterwarnings("ignore")

import lucid.models  # noqa: F401  — populates the weights registry
import lucid.weights as weights_mod
from lucid.models import create_model
from lucid.nn import Module
from lucid.weights import WeightEntry, WeightsEnum
from lucid.models._registry import _REGISTRY
from lucid.weights._registry import _WEIGHTS_BY_MODEL

_TIMEOUT = 90


def _normalise(shape: object) -> tuple[int, ...]:
    """Treat a 0-d scalar and a length-1 vector as the same thing.

    ``num_batches_tracked`` is a scalar in the model and is written as
    ``[1]`` by the safetensors converter.  ``load_state_dict`` accepts
    either, so a check that did not would flag every BatchNorm model.
    """
    dims = tuple(shape)  # type: ignore[call-overload]
    return () if dims in ((), (1,)) else dims


def _header(url: str) -> dict[str, object]:
    """Read a safetensors header without fetching the tensors."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        size = struct.unpack("<Q", response.read(8))[0]
    request = urllib.request.Request(
        url, headers={"Range": f"bytes=8-{8 + size - 1}"}
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read(size).decode())


@contextmanager
def _load_intercepted() -> Iterator[list[tuple[Module, WeightEntry]]]:
    """Run factories with the download-and-load step replaced by a note.

    What the factory passes here is the ground truth this check needs:
    the module that actually receives the checkpoint, after every
    config decision the factory made.
    """
    seen: list[tuple[Module, WeightEntry]] = []
    original = weights_mod.load_weight_entry

    def spy(
        model: Module,
        weights: WeightsEnum | WeightEntry,
        *,
        name: str,
        strict: bool = True,
    ) -> object:
        entry = weights.entry if isinstance(weights, WeightsEnum) else weights
        seen.append((model, entry))
        return None

    weights_mod.load_weight_entry = spy  # type: ignore[assignment]
    try:
        yield seen
    finally:
        weights_mod.load_weight_entry = original  # type: ignore[assignment]


def _tags(enum: type[WeightsEnum]) -> list[str]:
    """Every real member, with ``DEFAULT`` folded into what it aliases."""
    return [tag for tag in enum.__members__ if tag != "DEFAULT"]


def _fit(model_name: str, tag: str) -> dict[str, object]:
    """Compare one checkpoint against the module the factory loads it into."""
    with _load_intercepted() as seen:
        try:
            create_model(model_name, pretrained=tag)
        except ValueError as exc:
            if not seen:
                # A factory may share an enum with a sibling and refuse the
                # sibling's tags — diamond and diamond_csgo do, because the
                # registry maps a factory to an enum and not to a subset of
                # its tags.  Refusing is the correct answer, so it is not a
                # failure; but it must be recorded, or a checkpoint every
                # factory declines would vanish from the report.
                return {"declined": str(exc)[:160]}
            return {"error": f"ValueError: {exc}"[:200]}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    if not seen:
        return {"error": "the factory did not load any weights for this tag"}
    target, entry = seen[-1]

    try:
        header = _header(entry.url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"unreachable": f"{type(exc).__name__}: {exc}"[:160]}

    stored = {
        entry.key_map.get(key, key): _normalise(value["shape"])  # type: ignore[index]
        for key, value in header.items()
        if key != "__metadata__"
    }
    wanted = {k: _normalise(v.shape) for k, v in target.state_dict().items()}

    return {
        "tensors": len(stored),
        "shape_mismatch": sorted(
            k for k in stored if k in wanted and wanted[k] != stored[k]
        ),
        "checkpoint_only": sorted(set(stored) - set(wanted)),
        "model_only": sorted(set(wanted) - set(stored)),
    }


def _describe(model_name: str, tag: str, report: dict[str, object]) -> list[str]:
    """One line per thing wrong, naming enough to act on."""
    lines: list[str] = []
    where = f"{model_name} [{tag}]"
    for kind, label in (
        ("shape_mismatch", "shape differs"),
        ("checkpoint_only", "in the checkpoint, not the model"),
        ("model_only", "in the model, no weights for it"),
    ):
        keys = report.get(kind) or []
        assert isinstance(keys, list)
        if keys:
            shown = ", ".join(keys[:4])
            more = f" (+{len(keys) - 4} more)" if len(keys) > 4 else ""
            lines.append(f"  {where}: {len(keys)} {label} — {shown}{more}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--family",
        help="only this family, as registered (e.g. sknet, not sk_resnet)",
    )
    parser.add_argument("--model", help="only this factory")
    parser.add_argument(
        "--list", action="store_true", help="print what would be checked, then stop"
    )
    args = parser.parse_args()

    targets = sorted(_WEIGHTS_BY_MODEL.items())
    if args.model:
        targets = [(n, e) for n, e in targets if n == args.model]
        if not targets:
            print(f"no registered weights for {args.model!r}", file=sys.stderr)
            return 1
    elif args.family:
        # A factory's family is registry metadata, not a name prefix:
        # the sknet family ships sk_resnet_18_cls, which shares no
        # substring with it.
        targets = [
            (n, e)
            for n, e in targets
            if getattr(_REGISTRY.get(n), "family", None) == args.family
        ]
        if not targets:
            known = sorted(
                {
                    fam
                    for n in _WEIGHTS_BY_MODEL
                    if (fam := getattr(_REGISTRY.get(n), "family", None))
                }
            )
            print(
                f"no family {args.family!r} ships weights. Families that do:\n"
                f"  {', '.join(known)}",
                file=sys.stderr,
            )
            return 1

    checkpoints = sum(len(_tags(enum)) for _, enum in targets)
    if not checkpoints:
        # Reporting "OK" for an empty selection would let a typo'd
        # filter pass a CI job that checked nothing.
        print("no checkpoints selected — nothing was checked", file=sys.stderr)
        return 1
    if args.list:
        for name, enum in targets:
            print(f"{name}: {', '.join(_tags(enum))}")
        print(f"\n{len(targets)} factories, {checkpoints} checkpoints")
        return 0

    print(f"checking {checkpoints} checkpoints across {len(targets)} factories")
    bad: list[str] = []
    unreachable: list[str] = []
    checked = 0
    declined = 0
    # A checkpoint is fine if *some* factory can load it; one that every
    # factory declines is orphaned, and a refusal added by mistake would
    # otherwise hide it.
    fitted: set[str] = set()
    offered: set[str] = set()

    for index, (name, enum) in enumerate(targets, 1):
        # A run over every checkpoint takes tens of minutes, nearly all of
        # it waiting on range requests.  Without a line per factory there
        # is no way to tell a slow run from a hung one, which matters most
        # in CI, where nobody can look at the process.
        print(f"  [{index}/{len(targets)}] {name}", flush=True)
        for tag in _tags(enum):
            offered.add(tag)
            report = _fit(name, tag)
            if "declined" in report:
                declined += 1
                continue
            if "unreachable" in report:
                unreachable.append(f"  {name} [{tag}]: {report['unreachable']}")
                fitted.add(tag)  # unknown, not orphaned
                continue
            if "error" in report:
                bad.append(f"  {name} [{tag}]: {report['error']}")
                continue
            checked += 1
            problems = _describe(name, tag, report)
            if not problems:
                fitted.add(tag)
            bad.extend(problems)

    for tag in sorted(offered - fitted):
        bad.append(f"  [{tag}]: no factory in this run could load this checkpoint")

    print(f"{checked} checkpoints read", end="")
    print(f", {declined} declined by a sibling factory" if declined else "")
    if unreachable:
        print(f"\n{len(unreachable)} could not be reached:", file=sys.stderr)
        for line in unreachable:
            print(line, file=sys.stderr)
    if bad:
        print(f"\n{len(bad)} problem(s):", file=sys.stderr)
        for line in bad:
            print(line, file=sys.stderr)
        print(
            "\nA checkpoint that does not fit means pretrained=True raises for "
            "that factory. Either the factory's config drifted from the "
            "checkpoint, or the entry needs a key_map.",
            file=sys.stderr,
        )
        return 1

    print("[check_weight_fit] OK — every checkpoint fits its factory.")
    return 2 if unreachable else 0


if __name__ == "__main__":
    sys.exit(main())
