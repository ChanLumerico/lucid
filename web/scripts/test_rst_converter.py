#!/usr/bin/env python3
"""Regression tests for the reST -> markdown converter in build-api-data.py.

The converter (``_rst_to_text`` and friends) is regex-based and has a history
of silent leaks — every bug below was a real artifact that reached the rendered
docs site before the contract audit caught it.  These tests pin the fixed
behavior so a regex tweak can't regress it without a fast, build-free signal.

Standalone by design: the docs CI runner installs no Python packages (no
pytest), so this runs as ``python3 web/scripts/test_rst_converter.py`` and
exits non-zero on the first failure.  It is also pytest-collectable (plain
``test_*`` functions with ``assert``) for local ``pytest`` runs.
"""

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load_converter():
    """Import build-api-data.py (hyphenated filename) and return its module."""
    path = _HERE / "build-api-data.py"
    spec = importlib.util.spec_from_file_location("_build_api_data", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # guarded by `if __name__ == "__main__"`
    return mod


_M = _load_converter()
rst = _M._rst_to_text


def test_role_markup_stripped() -> None:
    """``:func:`x``` / ``:class:`X``` cross-reference roles lose the role
    prefix but keep the inline-code backticks — no raw ``:role:`` leak."""
    out = rst("See :func:`lucid.add` and :class:`Tensor`.")
    assert ":func:" not in out and ":class:" not in out, out
    assert "`lucid.add`" in out and "`Tensor`" in out, out


def test_literal_block_double_colon_collapses() -> None:
    """A trailing ``::`` literal-block marker collapses to a single ``:``;
    the indented block survives."""
    out = rst("Example::\n\n    x = add(a, b)")
    assert "Example::" not in out, out
    assert "Example:" in out and "x = add(a, b)" in out, out


def test_doctest_blocks_are_fenced() -> None:
    """``>>>`` doctest lines get wrapped in a ```python fence (never left as a
    reST blockquote / bare prose)."""
    out = rst("Usage:\n\n>>> import lucid\n>>> lucid.add(1, 2)")
    assert "```python" in out, out
    assert out.count("```") >= 2, out  # opened + closed
    assert ">>> import lucid" in out, out


def test_rst_hyperlink_to_markdown() -> None:
    """reST ``\\`text <url>\\`_`` anonymous hyperlinks become ``[text](url)``
    with no trailing ``\\`_`` or angle-bracket URL leak."""
    out = rst("See `the paper <https://arxiv.org/abs/1234>`_ for details.")
    assert "[the paper](https://arxiv.org/abs/1234)" in out, out
    assert "`_" not in out and "<https://" not in out, out


def test_inline_literal_double_to_single_backtick() -> None:
    """reST ``\\`\\`x\\`\\``` inline literals render as single-backtick code."""
    out = rst("Pass ``axis=0`` to reduce.")
    assert "`axis=0`" in out, out
    assert "``axis=0``" not in out, out


def test_inline_math_preserved() -> None:
    """``$...$`` math is passed through untouched for KaTeX (inline markup
    must not corrupt the TeX)."""
    out = rst(r"The norm $\|x\|_2$ is computed.")
    assert r"$\|x\|_2$" in out, out


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc!r}", file=sys.stderr)
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} converter tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
