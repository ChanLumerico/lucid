"""The ratchet gate fails on each seeded kind of slop and only ever lowers its baseline.

Each end-to-end case builds a scratch git repository with a small ``lucid/``
tree, writes its baseline with ``--rebaseline``, commits, seeds one
regression and runs the gate the way the hook and land do — as a process.

Not under ``lucid/test`` (the default ``testpaths``), so it runs on request::

    .venv/bin/python3 -m pytest tools/test_quality_gate.py
"""

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_gate.collectors.counters import count_python
from tools.quality_gate.core import Baseline, MergeRefused, merge_baselines
from tools.quality_gate.unlaunder import strip_ignores, strip_transitional, unwrap_casts

REPO = Path(__file__).resolve().parents[1]
FAST = "ruff,counters,lizard"

CLEAN = '''\
def add(a: int, b: int) -> int:
    """Sum."""
    return a + b
'''

BLOCK = "".join(
    f"    total_{i} = values[{i}] * weights[{i}] + offsets[{i}] - bias_{i}\n" for i in range(12)
)
DUPLICATED = f"def first(values, weights, offsets, bias_0):\n{BLOCK}    return 0\n\n\n" + (
    f"def second(values, weights, offsets, bias_0):\n{BLOCK}    return 1\n"
)


# ── unit: counters ──────────────────────────────────────────────────────────


def test_counters_read_comments_and_calls_not_strings() -> None:
    src = (
        "from typing import cast\n"
        "x = cast(int, y)  # type: ignore[arg-type, misc]\n"
        "z = 1  # type: ignore\n"
        "s = 'cast(int, y)  # type: ignore[override]'\n"
        "w = f(1)  # noqa: BLE001, S110\n"
        "v = 2  # noqa\n"
    )
    assert count_python(src) == {
        "cast": 1,
        "type-ignore[arg-type]": 1,
        "type-ignore[misc]": 1,
        "type-ignore-bare": 1,
        "noqa[BLE001]": 1,
        "noqa[S110]": 1,
        "noqa-bare": 1,
    }


# ── unit: the merge driver ──────────────────────────────────────────────────

A = ("ruff", "B905", "lucid/a.py")
B = ("ruff", "B905", "lucid/b.py")
REBASE = {"date": "2026-10-05", "reason": "new collector", "by": "orchestrator"}


def _b(
    counts: dict[tuple[str, str, str], int], history: list[dict[str, str]] | None = None
) -> Baseline:
    return Baseline(dict(counts), rebaselines=list(history or []))


def test_merge_disjoint_lowerings_keep_both() -> None:
    merged = merge_baselines(_b({A: 5, B: 5}), _b({A: 3, B: 5}), _b({A: 5, B: 2}))
    assert merged.counts == {A: 3, B: 2}


def test_merge_same_key_lowered_on_both_sides_takes_the_lower() -> None:
    merged = merge_baselines(_b({A: 5}), _b({A: 3}), _b({A: 4}))
    assert merged.counts == {A: 3}


def test_merge_delete_beats_lower() -> None:
    merged = merge_baselines(_b({A: 5, B: 1}), _b({B: 1}), _b({A: 4, B: 1}))
    assert merged.counts == {B: 1}


def test_merge_refuses_an_unauthorised_raise() -> None:
    with pytest.raises(MergeRefused):
        merge_baselines(_b({A: 5}), _b({A: 6}), _b({A: 5}))
    with pytest.raises(MergeRefused):
        merge_baselines(_b({}), _b({}), _b({B: 1}))


def test_merge_keeps_a_raise_carried_by_a_rebaseline() -> None:
    merged = merge_baselines(_b({A: 5}), _b({A: 4}), _b({A: 5, B: 9}, [REBASE]))
    assert merged.counts == {A: 4, B: 9}
    assert merged.rebaselines == [REBASE]


# ── end to end ──────────────────────────────────────────────────────────────


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@t",
        PYTHONPATH=str(REPO),
    )
    return env


def gate(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.quality_gate", "--root", str(root), *args],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=root,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "lucid" / "pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "pkg" / "mod.py").write_text(CLEAN)
    (root / "lucid" / "pkg" / "other.py").write_text(CLEAN + "x = 1  # type: ignore[misc]\n")
    git(root, "init", "-q", "-b", "main")
    done = gate(root, "--rebaseline", "--reason", "test", "--collectors", FAST)
    assert done.returncode == 0, done.stdout + done.stderr
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


def _append(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + text)


SEEDS = {
    "type-ignore": "\ny: int = 'a'  # type: ignore[assignment]\n",
    "except-pass": "\ntry:\n    add(1, 2)\nexcept Exception:\n    pass\n",
    "long-function": "\n\ndef long() -> int:\n    n = 0\n"
    + "    n += 1\n" * 200
    + "    return n\n",
}


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_fast_check_fails_on_seeded_slop(repo: Path, seed: str) -> None:
    assert gate(repo, "--fast").returncode == 0
    _append(repo, "lucid/pkg/mod.py", SEEDS[seed])
    out = gate(repo, "--fast")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "rose above the baseline" in out.stdout


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_diff_reports_the_increase_against_a_ref(repo: Path, seed: str) -> None:
    _append(repo, "lucid/pkg/mod.py", SEEDS[seed])
    git(repo, "commit", "-qam", "slop")
    out = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "SLOP DELTA vs HEAD~1" in out.stdout
    assert "← INCREASE" in out.stdout


def test_new_file_may_not_start_over_the_loc_cap(repo: Path) -> None:
    (repo / "lucid" / "pkg" / "big.py").write_text("x = 1\n" * 1600)
    out = gate(repo, "--fast")
    assert out.returncode == 1
    assert "file-loc-over-cap" in out.stdout


def test_a_decrease_must_be_recorded_and_update_only_lowers(repo: Path) -> None:
    (repo / "lucid" / "pkg" / "other.py").write_text(CLEAN)
    out = gate(repo, "--fast")
    assert out.returncode == 1 and "out of date" in out.stdout
    slack = gate(repo, "--fast", "--allow-slack")
    assert slack.returncode == 0 and "higher than the code" in slack.stdout
    assert gate(repo, "--update", "--fast").returncode == 0
    assert gate(repo, "--fast").returncode == 0
    data = json.loads((repo / "tools" / "quality_baseline.json").read_text())
    assert "type-ignore[misc]" not in data["counts"].get("counters", {})
    # A raise is refused and leaves the baseline untouched.
    before = (repo / "tools" / "quality_baseline.json").read_text()
    _append(repo, "lucid/pkg/mod.py", SEEDS["type-ignore"])
    assert gate(repo, "--update", "--fast").returncode == 1
    assert (repo / "tools" / "quality_baseline.json").read_text() == before


def test_a_move_is_not_new_debt(repo: Path) -> None:
    git(repo, "mv", "lucid/pkg/other.py", "lucid/pkg/moved.py")
    out = gate(repo, "--fast")
    assert "rose above" not in out.stdout, out.stdout
    assert "moved: lucid/pkg/other.py -> lucid/pkg/moved.py" in out.stdout
    assert gate(repo, "--update", "--fast").returncode == 0
    assert gate(repo, "--fast").returncode == 0
    git(repo, "commit", "-qam", "move")
    diff = gate(repo, "--diff", "HEAD~1", "--collectors", FAST)
    assert diff.returncode == 0, diff.stdout


@pytest.mark.skipif(
    shutil.which("npx") is None and not Path("/opt/homebrew/bin/npx").exists(),
    reason="jscpd runs through npx",
)
def test_duplicated_block_fails_full_and_diff(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "lucid").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "lucid" / "a.py").write_text(CLEAN)
    git(root, "init", "-q", "-b", "main")
    assert gate(root, "--rebaseline", "--reason", "t", "--collectors", "jscpd").returncode == 0
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    (root / "lucid" / "a.py").write_text(DUPLICATED)
    for mode in ("--full", "--fast"):
        out = gate(root, mode, "--collectors", "jscpd")
        assert out.returncode == 1 and "clone-pair" in out.stdout, out.stdout + out.stderr
    git(root, "commit", "-qam", "dup")
    assert gate(root, "--diff", "HEAD~1", "--collectors", "jscpd").returncode == 1


def test_rebase_merges_concurrent_lowerings_through_the_driver(tmp_path: Path) -> None:
    """Two branches lower neighbouring keys; the rebase neither conflicts nor loses either."""
    root = tmp_path / "repo"
    (root / "lucid" / "pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    ignores = "".join(f"x{i} = 1  # type: ignore[misc]\n" for i in range(3))
    for name in ("a.py", "b.py"):
        (root / "lucid" / "pkg" / name).write_text(CLEAN + ignores)
    (root / ".gitattributes").write_text(
        "tools/quality_baseline.json merge=lucid-quality-baseline\n"
    )
    git(root, "init", "-q", "-b", "main")
    driver = f"{sys.executable} -m tools.quality_gate --merge-baseline %O %A %B"
    git(root, "config", "merge.lucid-quality-baseline.driver", driver)
    assert gate(root, "--rebaseline", "--reason", "t", "--collectors", FAST).returncode == 0
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")

    def lower(name: str, keep: int) -> None:
        kept = "x0 = 1  # type: ignore[misc]\n" * keep
        (root / "lucid" / "pkg" / name).write_text(CLEAN + kept)
        assert gate(root, "--update", "--collectors", FAST).returncode == 0
        git(root, "commit", "-qam", f"lower {name}")

    git(root, "checkout", "-q", "-b", "side")
    lower("a.py", 1)
    git(root, "checkout", "-q", "main")
    lower("b.py", 2)
    git(root, "checkout", "-q", "side")
    git(root, "rebase", "-q", "main")  # git() asserts it succeeded
    counts = json.loads((root / "tools" / "quality_baseline.json").read_text())["counts"]
    assert counts["counters"]["type-ignore[misc]"] == {"lucid/pkg/a.py": 1, "lucid/pkg/b.py": 2}
    assert gate(root, "--full", "--collectors", FAST).returncode == 0


# ── unlaunder: the mypy-driven codemod ──────────────────────────────────────


def test_unlaunder_strips_only_the_marked_sections() -> None:
    ini = (
        "[mypy]\nstrict = True\n\n"
        "# transitional (LCD-260) — removed later.\n# more words\n"
        "[mypy-lucid.pkg.*]\nwarn_unused_ignores = False\n\n"
        "[mypy-lucid.keep.*]\nignore_errors = True\n"
        "# transitional (LCD-260) is a stray marker here\nfoo = 1\n"
        "[mypy-lucid.also_kept]\nignore_errors = True\n"
    )
    out = strip_transitional(ini)
    assert "[mypy-lucid.pkg.*]" not in out and "warn_unused_ignores" not in out
    assert "[mypy-lucid.keep.*]" in out and "[mypy-lucid.also_kept]" in out
    assert "strict = True" in out


@pytest.mark.parametrize(
    ("line", "unused", "after"),
    [
        ("x = 1  # type: ignore\n", None, "x = 1\n"),
        ("x = 1  # type: ignore[misc]\n", None, "x = 1\n"),
        ("x = 1  # type: ignore[a, b]\n", {"a"}, "x = 1  # type: ignore[b]\n"),
        ("x = 1  # type: ignore[a, b]  # why\n", {"a"}, "x = 1  # type: ignore[b]  # why\n"),
        ("x = 1  # type: ignore[a, b]\n", {"a", "b"}, "x = 1\n"),
        ("x = 1  # type: ignore[misc]  # it is fine at runtime\n", None, "x = 1\n"),
        ("x = 1  # type: ignore # noqa: F821\n", None, "x = 1  # noqa: F821\n"),
        ("x = 1  # noqa: E501  # type: ignore[misc]\n", None, "x = 1  # noqa: E501\n"),
        ("s = '# type: ignore'  # type: ignore\n", None, "s = '# type: ignore'\n"),
    ],
)
def test_unlaunder_removes_only_the_unused_codes(
    line: str, unused: set[str] | None, after: str
) -> None:
    out, skipped = strip_ignores(line, {1: None if unused is None else frozenset(unused)})
    assert (out, skipped) == (after, [])


def test_unlaunder_leaves_an_ignore_whose_codes_do_not_match() -> None:
    out, skipped = strip_ignores("x = 1  # type: ignore[misc]\n", {1: frozenset({"override"})})
    assert out == "x = 1  # type: ignore[misc]\n" and len(skipped) == 1


def _sites(src: str) -> set[tuple[int, int]]:
    """Every cast call's (line, byte column), as mypy reports them."""
    return {
        (n.lineno, n.col_offset)
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "cast"
    }


@pytest.mark.parametrize(
    ("src", "after"),
    [
        ("y = cast(int, cast(int, x)) + cast(int, x)\n", "y = x + x\n"),
        ("y = cast(\n    int,\n    f(\n        x,\n    ),\n)\n", "y = f(\n        x,\n    )\n"),
        ("y = cast(int, a + b) * 2\n", "y = (a + b) * 2\n"),
        ("y = cast(int, a + b)\n", "y = a + b\n"),
        ("f(cast(int, a if b else c))\n", "f(a if b else c)\n"),
        ("y = cast(int, a or b).bit_length()\n", "y = (a or b).bit_length()\n"),
        ("y = cast(int, a\n    + b)\n", "y = (a\n    + b)\n"),
        ("f(cast(int, a\n    + b))\n", "f(a\n    + b)\n"),
        ("y = -cast(int, (z := 1))\n", "y = -(z := 1)\n"),
        ("y = typing.cast(int, x)\n", "y = x\n"),
        ("y = cast(typ=int, val=x)\n", "y = x\n"),
        ("s = 'é'; y = cast(int, x)\n", "s = 'é'; y = x\n"),
    ],
)
def test_unlaunder_unwraps_casts_without_changing_the_code(src: str, after: str) -> None:
    out, n, skipped = unwrap_casts(src, _sites(src))
    assert (out, skipped) == (after, [])
    assert n == len(_sites(src))


def test_unlaunder_leaves_a_cast_whose_comment_would_be_lost() -> None:
    src = "y = cast(\n    int,  # the engine returns int here\n    x,\n)\n"
    out, n, skipped = unwrap_casts(src, _sites(src))
    assert out == src and n == 0
    assert skipped == ["1: a comment inside the cast would be lost"]


def test_unlaunder_unwraps_only_the_reported_cast_of_a_nest() -> None:
    src = "y = cast(int, cast(str, x))\n"
    out, n, _ = unwrap_casts(src, {(1, 14)})
    assert (out, n) == ("y = cast(int, x)\n", 1)


UNLAUNDER_INI = """\
[mypy]
python_version = 3.14
strict = True

# transitional (LCD-260) — keeps the package green until its sweep lands.
[mypy-lucid.pkg.*]
warn_unused_ignores = False
disable_error_code = redundant-cast
"""

UNLAUNDER_MOD = """\
from typing import cast


def f(x: int, s: str) -> int:
    a = cast(int, cast(int, x)) + cast(int, x)
    b = cast(
        int,
        x,
    )
    c: int = s  # type: ignore[assignment, misc]
    d = 1  # type: ignore[misc]  # the reason this was needed
    e = 2  # type: ignore # noqa: E501
    return a + b + c + d + e
"""

UNLAUNDER_DONE = """\
def f(x: int, s: str) -> int:
    a = x + x
    b = x
    c: int = s  # type: ignore[assignment]
    d = 1
    e = 2  # noqa: E501
    return a + b + c + d + e
"""


def unlaunder(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tools.quality_gate.unlaunder", "--root", str(root), *args],
        cwd=REPO,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("black") is None and shutil.which("black") is None,
    reason="unlaunder reformats with black",
)
def test_unlaunder_end_to_end_reaches_a_fixpoint(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for pkg in ("lucid/pkg", "lucid/other"):
        (root / pkg).mkdir(parents=True)
        (root / pkg / "__init__.py").write_text("")
    (root / "lucid" / "__init__.py").write_text("")
    (root / "mypy.ini").write_text(UNLAUNDER_INI)
    (root / "lucid" / "pkg" / "mod.py").write_text(UNLAUNDER_MOD)
    (root / "lucid" / "other" / "mod.py").write_text(UNLAUNDER_MOD)

    # A path mypy never reports on would read as "0 left": refused instead.
    (root / "tools").mkdir()
    (root / "lucid" / "test").mkdir()
    for wrong in ("lucid/nope", "tools", "lucid/test", str(tmp_path)):
        assert unlaunder(root, "--check", "--paths", wrong).returncode == 2, wrong
    check = unlaunder(root, "--check", "--paths", str(root / "lucid" / "pkg"))
    assert check.returncode == 1, check.stdout + check.stderr
    assert "2 redundant-cast, 3 unused-ignore" in check.stdout  # one per line
    assert (root / "lucid" / "pkg" / "mod.py").read_text() == UNLAUNDER_MOD

    run = unlaunder(root, "--paths", "lucid/pkg")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "round 2:" in run.stdout  # mypy reports one cast per line and message
    assert (root / "lucid" / "pkg" / "mod.py").read_text() == UNLAUNDER_DONE
    assert (root / "lucid" / "other" / "mod.py").read_text() == UNLAUNDER_MOD
    done = unlaunder(root, "--check", "--paths", "lucid/pkg")
    assert done.returncode == 0 and "0 redundant-cast, 0 unused-ignore" in done.stdout
