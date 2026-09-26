"""The public surface of the last release, and what may not change about it.

Semantic Versioning promises that code written against Lucid 3.x keeps
running on every later 3.x.  Nothing held Lucid to it: minor releases
renamed zoo factories (``mobilenet_v1`` became ``mobilenet``), dropped 43
names from ``lucid.models`` and changed constructors, and the only notice
was a CHANGELOG line.

``released_surface.json`` records, for every public name in the last
release, the parameters a caller could pass: each one's name, whether it
went by position, by keyword or either, and whether it could be left out.
A later tree breaks a caller when one of those calls stops binding — the
name gone, a parameter renamed, moved, made keyword-only or made required.
Adding a name or an optional parameter breaks nobody and passes; so does
changing a default or an annotation, which no call site spells.  A name
marked with :func:`lucid._deprecation.deprecated` may leave once the release
it was marked to leave in has come — and so may everything under it.

Only the release commit rewrites the snapshot, after bumping the version::

    python -m lucid.test.audit._released --update

which refuses while anything released would break.
"""

import annotationlib
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lucid
from lucid._deprecation import _version
from lucid.test.audit import _surface

#: The surface as the last release shipped it.
SNAPSHOT = Path(__file__).with_name("released_surface.json")

_KIND = {
    inspect.Parameter.POSITIONAL_ONLY: "p",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "a",
    inspect.Parameter.VAR_POSITIONAL: "v",
    inspect.Parameter.KEYWORD_ONLY: "k",
    inspect.Parameter.VAR_KEYWORD: "w",
}

Surface = dict[str, dict[str, Any]]


def _unwrap(obj: object) -> object:
    return obj.__func__ if isinstance(obj, staticmethod | classmethod) else obj


def _params(obj: object) -> list[str] | None:
    """``["a:input", "a:dim=", "k:keepdim="]`` — kind, name, optional."""
    obj = _unwrap(obj)
    if isinstance(obj, property) or not callable(obj):
        return None
    try:
        signature = inspect.signature(
            obj, annotation_format=annotationlib.Format.FORWARDREF
        )
    except ValueError, TypeError:
        return None  # a builtin: its existence is still held
    return [
        f"{_KIND[p.kind]}:{p.name}{'' if p.default is p.empty else '='}"
        for p in signature.parameters.values()
    ]


def _entry(obj: object) -> dict[str, Any]:
    entry: dict[str, Any] = {"params": _params(obj)}
    marks = getattr(_unwrap(obj), "__dict__", {}).get("__lucid_deprecation__")
    if marks:
        entry["deprecated"] = dict(marks)
    return entry


def _members(qualname: str, cls: type, out: Surface) -> None:
    """A class's own public methods — ``forward``, ``step``, ``state_dict``."""
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or isinstance(
            member, staticmethod | classmethod | property
        ):
            out[f"{qualname}.{name}"] = _entry(member)


def _add(qualname: str, obj: object, out: Surface) -> None:
    out[qualname] = _entry(obj)
    # ``Tensor``'s methods are enumerated by the audit itself, as ``Tensor.*``.
    if (
        inspect.isclass(obj)
        and obj.__module__.startswith("lucid.")
        and obj is not lucid.Tensor
    ):
        _members(qualname, obj, out)


def collect() -> Surface:
    """Every public name and what a caller could pass it, today."""
    out: Surface = {}
    for symbol in _surface.enumerate_surface():
        _add(symbol.qualname, symbol.obj, out)
    # The zoo is outside the audit's surface but not outside the promise:
    # its factory names are what user code spells most often.
    import lucid.models as models

    for name in sorted(models.__all__):
        _add(f"lucid.models.{name}", getattr(models, name), out)
    return dict(sorted(out.items()))


def _split(params: list[str]) -> list[tuple[str, str, bool]]:
    return [(token[0], token[2:].rstrip("="), token.endswith("=")) for token in params]


def incompatible(old: list[str] | None, new: list[str] | None) -> str | None:
    """Why a call that bound to ``old`` may no longer bind to ``new``."""
    if old is None or new is None:
        return None
    before, after = _split(old), _split(new)
    takes_args = any(kind == "v" for kind, _, _ in after)
    takes_kwargs = any(kind == "w" for kind, _, _ in after)
    old_positional = [p for p in before if p[0] in "pa"]
    new_positional = [p for p in after if p[0] in "pa"]
    by_keyword = {name: optional for kind, name, optional in after if kind in "ak"}

    for index, (kind, name, optional) in enumerate(old_positional):
        if index >= len(new_positional):
            if not takes_args:
                return f"parameter {name!r} can no longer be passed by position"
            if kind == "a" and name not in by_keyword and not takes_kwargs:
                return f"parameter {name!r} can no longer be passed by keyword"
            continue
        new_kind, new_name, new_optional = new_positional[index]
        if kind == "a" and (new_kind != "a" or new_name != name):
            return f"parameter {name!r} was renamed, moved or made positional-only"
        if optional and not new_optional:
            return f"parameter {name!r} is required now"
    for _, name, optional in new_positional[len(old_positional) :]:
        if not optional:
            return f"new parameter {name!r} is required"

    old_names = {name for kind, name, _ in before if kind in "ak"}
    for kind, name, optional in before:
        if kind != "k":
            continue
        if name not in by_keyword:
            if not takes_kwargs:
                return f"keyword {name!r} is no longer accepted"
        elif optional and not by_keyword[name]:
            return f"keyword {name!r} is required now"
    for kind, name, optional in after:
        if kind == "k" and not optional and name not in old_names:
            return f"new keyword {name!r} is required"

    if any(kind == "v" for kind, _, _ in before) and not takes_args:
        return "extra positional arguments are no longer accepted"
    if any(kind == "w" for kind, _, _ in before) and not takes_kwargs:
        return "extra keyword arguments are no longer accepted"
    return None


def _removal(name: str, released: Mapping[str, dict[str, Any]]) -> str | None:
    """The release ``name`` — or a name it lives under — was marked to leave in."""
    parts = name.split(".")
    for end in range(len(parts), 0, -1):
        marks = released.get(".".join(parts[:end]), {}).get("deprecated")
        if marks:
            return str(marks["removal"])
    return None


def breaks(
    released: Mapping[str, dict[str, Any]], current: Surface, version: str
) -> list[str]:
    """Every released call that ``current`` would stop accepting at ``version``."""
    now = _version(version)
    problems = []
    for name, entry in released.items():
        if name not in current:
            removal = _removal(name, released)
            if removal is not None and now >= _version(removal):
                continue
            problems.append(
                f"{name}: removed before {removal}, the release it was deprecated until"
                if removal
                else f"{name}: removed without a deprecation"
            )
            continue
        why = incompatible(entry["params"], current[name]["params"])
        if why:
            problems.append(f"{name}: {why}")
    return problems


def load() -> dict[str, Any]:
    return dict(json.loads(SNAPSHOT.read_text()))


def _write(version: str, surface: Surface) -> None:
    # One name per line, so a release's diff reads as the API it changed.
    lines = [
        f"  {json.dumps(name)}: {json.dumps(entry)}" for name, entry in surface.items()
    ]
    SNAPSHOT.write_text(
        '{\n"version": '
        + json.dumps(version)
        + ',\n"symbols": {\n'
        + ",\n".join(lines)
        + "\n}}\n"
    )


def main(argv: list[str]) -> int:
    if argv != ["--update"]:
        print("usage: python -m lucid.test.audit._released --update", file=sys.stderr)
        return 2
    current = collect()
    if SNAPSHOT.exists():
        problems = breaks(load()["symbols"], current, lucid.__version__)
        if problems:
            print(f"{len(problems)} released call(s) would break:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
    _write(lucid.__version__, current)
    print(f"{SNAPSHOT.name}: {len(current)} names at {lucid.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
