"""H1, H5, H6 and H7 hold across the whole ``lucid/`` tree, tests included.

The runtime package was clean, but the test tree was not. It held 14 mentions
of the reference framework's packages by name and 401 string annotations, and
nothing checked either rule. They surfaced only when closed issues were re-read
(Linear CHA-15, CHA-16), so this does the checking.  After that, tests that
imported the reference framework directly and opened with
``from __future__ import annotations`` landed anyway, because nothing ran the
check before a commit.  The rules now live in ``tools/check_source_rules.py``,
which ``.githooks/pre-commit`` runs on the staged files and this runs on the
whole tree.

* **H1** — no ``from __future__ import annotations``.
* **H5** — the reference framework is imported and named only in
  :mod:`lucid.test._fixtures.ref_framework`; every other test takes the
  ``ref`` fixture.  The tool reads the name from that module.
* **H6** — the other vendor's GPU API is not named anywhere.
* **H7** — no annotation is a string (PEP 649 makes the quotes unnecessary).
"""

import functools

import pytest

from lucid.test._fixtures import ref_framework
from tools import check_source_rules as rules


@functools.cache
def _tree() -> tuple[rules.Violation, ...]:
    return tuple(rules.check(rules.tree_sources(), rules.ref_name()))


def _hits(*names: str) -> list[str]:
    return [str(v) for v in _tree() if v.rule in names]


def test_the_tool_reads_the_fixture_s_name() -> None:
    assert rules.ref_name() == ref_framework._REF_NAME


def test_h1_no_future_annotations_import() -> None:
    hits = _hits("future-annotations", "syntax")
    assert not hits, "H1 — drop the import (PEP 649):\n" + "\n".join(hits)


def test_h5_only_the_fixture_imports_the_reference_framework() -> None:
    hits = _hits("ref-import")
    assert not hits, "H5 — take the ref fixture:\n" + "\n".join(hits)


def test_h5_the_reference_framework_is_named_in_one_file() -> None:
    hits = _hits("ref-name")
    assert not hits, "H5 — use the ref fixture or 'reference framework':\n" + "\n".join(
        hits
    )


def test_h6_no_other_gpu_api_is_named() -> None:
    hits = _hits("gpu-name")
    assert not hits, "H6 — Lucid's GPU is 'metal':\n" + "\n".join(hits)


def test_h7_no_annotation_is_a_string() -> None:
    hits = _hits("string-annotation")
    assert not hits, "H7 — drop the quotes (PEP 649):\n" + "\n".join(hits)


_NAME = ref_framework._REF_NAME
_GPU = "cu" + "da"

# (source, the rule it breaks).  Each is checked as a test file and must be
# reported; the clean twins below must not be.
_BROKEN = [
    ("from __future__ import annotations\n", "future-annotations"),
    (f"import {_NAME}\n", "ref-import"),
    (f"import {_NAME}.nn as nn\n", "ref-import"),
    (f"from {_NAME} import nn\n", "ref-import"),
    (f"import {_NAME}vision\n", "ref-import"),
    (f"mod = pytest.importorskip({_NAME!r})\n", "ref-import"),
    (f"mod = importlib.import_module({_NAME!r})\n", "ref-import"),
    (f"# matches {_NAME.capitalize()}'s rounding\n", "ref-name"),
    (f"x = 1  # the {_GPU.upper()} path\n", "gpu-name"),
    ("def f(x: 'Tensor') -> None: ...\n", "string-annotation"),
]
_CLEAN = [
    "from __future__ import division\n",
    "def f(ref: object) -> None:\n    ref.tensor([1.0])\n",
    "def f(mode: Literal['a', 'b']) -> None: ...\n",
    "x: Annotated[int, 'metadata'] = 1\n",
    "# the reference framework rounds to even\n",
]


@pytest.mark.parametrize("source,rule", _BROKEN, ids=[r for _, r in _BROKEN])
def test_each_rule_reports_its_violation(source: str, rule: str) -> None:
    found = rules.check([("lucid/test/unit/test_x.py", source)], _NAME)
    assert rule in {v.rule for v in found}, found


@pytest.mark.parametrize("source", _CLEAN)
def test_clean_source_is_not_reported(source: str) -> None:
    assert rules.check([("lucid/test/unit/test_x.py", source)], _NAME) == []


def test_the_fixture_may_name_and_import_the_reference_framework() -> None:
    source = f"import {_NAME}\n_REF_NAME = {_NAME!r}\n"
    assert rules.check([(rules.REF_FILE, source)], _NAME) == []


def test_files_outside_lucid_are_not_checked() -> None:
    assert rules.check([("tools/x.py", f"import {_NAME}\n")], _NAME) == []
