#!/usr/bin/env python3
"""tools/check_dead_config_fields.py — a Config field the model never reads.

A model family's ``_config.py`` is a frozen dataclass, and nothing
checks that the fields it declares reach the forward pass. They
routinely do not: the config is usually written before the model, and a
field that never gets wired changes no shape, raises nothing and is
invisible to every other gate.

V-JEPA 2 shipped with four of them. ``drop_path_rate`` was the clearest:

    rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
    blocks = [_Block(...) for _ in rates]     # rates counted, never used

The per-depth rates were computed and thrown away, ``_Block`` had no
DropPath at all, and the validator, the family contract test, ``mypy
--strict``, ruff and the summary builder all passed. A user setting
``drop_path_rate=0.3`` got a model identical to ``0.0``.

This is deliberately a *name* check, not a dataflow one: it asks
whether the field's name appears anywhere in the family's own sources
apart from the config that declares it — ``_model.py`` reads most of
them, but a factory legitimately consumes ``variant`` or
``rpn_nms_thresh`` in ``_pretrained.py`` and never in the model. That
is cheap and has no false negatives worth worrying about — a field the
model genuinely uses has to name it somewhere. It does have false
positives, which is what ``_ALLOWED`` is for, and it cannot see the
other half of the failure: a name that *is* mentioned but is overridden
at the call site, as ``uniform_power`` was by ``_sincos_3d(dim, grid,
False)``. Read the call sites when adding a family; this catches the
half that a reader would otherwise have to hold in their head.

Run::

    python tools/check_dead_config_fields.py
    python tools/check_dead_config_fields.py --list   # every field, with its verdict

Seventeen fields across thirteen families were already unreferenced
when this check was written. They are listed in ``_BASELINE``, which is
not an excuse list: the check still prints them every run, and the gate
only fails on a field that is in neither ``_ALLOWED`` nor the baseline.
The point is to stop the eighteenth, not to claim the seventeen are
fine. Work one off the baseline and delete its line.

Exit codes
----------
0 — every declared field is named, allowed, or on the baseline.
1 — at least one is none of those.
"""

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = REPO_ROOT / "lucid" / "models"

#: Fields that are declared and legitimately not named in ``_model.py``,
#: each with the reason.  Keep this list short and each entry answerable:
#: an unexplained entry is how a real dead field hides.
_ALLOWED: dict[tuple[str, str], str] = {
    ("*", "model_type"): (
        "ClassVar the registry reads, not a family-local parameter"
    ),
    ("vjepa2_ac", "num_frames"): (
        "records the released clip length and sizes token_grid; the action "
        "model takes any frame count of two or more, which its config "
        "docstring says"
    ),
    ("vjepa", "sampling_rate"): (
        "records the released clip's frame stride; the model is handed an "
        "already-sampled clip"
    ),
}


#: Unreferenced when this check was introduced (2026-09-22), untriaged.
#: Each is either a real dead field or a decorative one that nobody has
#: written the reason for yet.  Shrinking this list is the work; the
#: gate exists to keep it from growing.
_BASELINE: frozenset[tuple[str, str]] = frozenset(
    {
        ("ddpm", "clip_denoised"),
        ("diamond", "burn_in"),
        ("flow_matching", "exact_trace_max_dim"),
        ("genie", "action_encoder_head_dim"),
        ("genie", "action_decoder_head_dim"),
        ("rectified_flow", "exact_trace_max_dim"),
        ("bert", "position_embedding_type"),
        ("gpt", "pad_token_id"),
        ("gpt2", "pad_token_id"),
        ("gpt2", "bos_token_id"),
        ("gpt2", "eos_token_id"),
        ("roformer", "position_embedding_type"),
        ("densenet", "memory_efficient"),
        ("mask_rcnn", "rpn_nms_thresh"),
        ("mask_rcnn", "rpn_min_size"),
        ("mask_rcnn", "rpn_score_thresh"),
        ("maskformer", "backbone_block"),
    }
)


def _declared_fields(config_path: Path) -> list[str]:
    """Public annotated names on every class in a family's ``_config.py``."""
    tree = ast.parse(config_path.read_text())
    names: list[str] = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in cls.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    return names


def _families() -> list[tuple[str, Path]]:
    """Every family directory holding both a config and a model."""
    found = []
    for config in sorted(MODELS_ROOT.rglob("_config.py")):
        model = config.with_name("_model.py")
        if model.is_file():
            found.append((config.parent.name, config.parent))
    return found


def _allowed(family: str, field: str) -> str | None:
    return _ALLOWED.get((family, field)) or _ALLOWED.get(("*", field))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--list", action="store_true", help="print every field and its verdict"
    )
    parser.add_argument("--family", help="only this family")
    args = parser.parse_args()

    dead: list[tuple[str, str]] = []
    known: list[tuple[str, str]] = []
    checked = 0
    for family, family_dir in _families():
        if args.family and family != args.family:
            continue
        # Every .py in the family except the config itself: a factory
        # that selects a variant reads the field and the model never
        # sees it, which is not the failure this looks for.
        source = [
            f.read_text()
            for f in sorted(family_dir.rglob("*.py"))
            if f.name != "_config.py"
        ]
        for field in _declared_fields(family_dir / "_config.py"):
            checked += 1
            if any(field in text for text in source):
                if args.list:
                    print(f"  used     {family}.{field}")
                continue
            reason = _allowed(family, field)
            if reason is not None:
                if args.list:
                    print(f"  allowed  {family}.{field} — {reason}")
                continue
            if (family, field) in _BASELINE:
                known.append((family, field))
                if args.list:
                    print(f"  baseline {family}.{field}")
                continue
            dead.append((family, field))
            if args.list:
                print(f"  DEAD     {family}.{field}")

    if known:
        print(
            f"[check_dead_config_fields] {len(known)} field(s) on the untriaged "
            f"baseline: " + ", ".join(f"{f}.{n}" for f, n in sorted(known))
        )

    if dead:
        print(
            f"[check_dead_config_fields] {len(dead)} field(s) declared but "
            f"named nowhere in their family outside the config:",
            file=sys.stderr,
        )
        for family, field in dead:
            print(f"  {family}.{field}", file=sys.stderr)
        print(
            "\n  Either wire the field into the forward pass, or delete it. If it "
            "is genuinely decorative, add it to _ALLOWED in this file with the "
            "reason — an entry nobody can answer is how a real one hides. Do "
            "not add it to _BASELINE; that list is closed and only shrinks.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[check_dead_config_fields] OK — {checked} field(s) across "
        f"{len(_families())} families, none newly unaccounted for."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
