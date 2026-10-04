"""The ratchet gate fails on each seeded kind of slop and only ever lowers its baseline.

Each end-to-end case builds a scratch git repository with a small ``lucid/``
tree, writes its baseline with ``--rebaseline``, commits, seeds one
regression and runs the gate the way the hook and land do — as a process.

Not under ``lucid/test`` (the default ``testpaths``), so it runs on request::

    .venv/bin/python3 -m pytest tools/test_quality_gate.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_gate.collectors.counters import count_python
from tools.quality_gate.core import Baseline, MergeRefused, merge_baselines

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
    out = gate(root, "--full", "--collectors", "jscpd")
    assert out.returncode == 1 and "clone-pair" in out.stdout, out.stdout + out.stderr
    git(root, "commit", "-qam", "dup")
    assert gate(root, "--diff", "HEAD~1", "--collectors", "jscpd").returncode == 1
