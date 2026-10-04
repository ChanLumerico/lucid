"""The claim gate judges every tree by that tree's own index and config.

git exports repository-local variables to the hooks it runs: GIT_INDEX_FILE
on every commit (relative ``.git/index`` in the main checkout, a temporary
``next-index-*.lock`` for ``git commit -- <path>``) and GIT_DIR from a linked
worktree.  A gate that hands them on to its queries of the *other* trees reads
the committing tree's index and labels there instead, and refuses every commit
with the committing tree named as the owner.  So these cases drive real commits
through a pre-commit hook in a scratch repository — main plus two linked
worktrees — once staged with ``git add`` and once as a partial commit, which is
also what proves the gate still reads the commit's own staged set from the
index git handed it.

Not under ``lucid/test`` (the default ``testpaths``), so it runs on request::

    .venv/bin/python3 -m pytest tools/test_agent_ws.py
"""

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

AGENT_WS = Path(__file__).resolve().with_name("agent_ws.py")
MODES = ("staged", "partial")


@dataclass
class Scratch:
    main: Path
    a: Path
    b: Path
    env: dict[str, str] = field(repr=False)

    def git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def edit(self, tree: Path, name: str) -> None:
        path = tree / name
        path.write_text(path.read_text() + f"{tree.name}\n")

    def commit(
        self, tree: Path, name: str, mode: str
    ) -> subprocess.CompletedProcess[str]:
        """Change ``name`` in ``tree`` and commit just that file through the hook."""
        self.edit(tree, name)
        message = f"{tree.name}: {name}"
        if mode == "staged":
            assert self.git(tree, "add", name).returncode == 0
            return self.git(tree, "commit", "-q", "-m", message)
        # `git commit -- <path>` builds a temporary index and hands it to the
        # hook as GIT_INDEX_FILE; the worktree's own index has nothing staged.
        return self.git(tree, "commit", "-q", "-m", message, "--", name)


@pytest.fixture
def scratch(tmp_path: Path) -> Scratch:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GIT_") and k != "LUCID_SKIP_CLAIMS"
    }
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        LUCID_AGENT_STATE=str(tmp_path / "state"),
    )
    main = tmp_path / "main"
    main.mkdir()
    scratch = Scratch(main, tmp_path / "wt-a", tmp_path / "wt-b", env)

    def run(cwd: Path, *args: str) -> None:
        proc = subprocess.run(
            list(args), cwd=cwd, env=env, capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, f"{args}: {proc.stderr}"

    run(main, "git", "init", "-q", "-b", "main")
    run(main, "git", "config", "user.name", "scratch")
    run(main, "git", "config", "user.email", "scratch@example.invalid")
    # As in the Lucid repository: `agent_ws.py label` writes `--worktree`
    # config, which is per worktree only with this extension on.
    run(main, "git", "config", "extensions.worktreeConfig", "true")
    for name in ("a.txt", "b.txt", "c.txt"):
        (main / name).write_text(f"{name}\n")
    run(main, "git", "add", ".")
    run(main, "git", "commit", "-q", "-m", "init")

    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(AGENT_WS))} check-commit\n"
    )
    hook.chmod(0o755)
    run(main, "git", "config", "core.hooksPath", str(hooks))

    for tree, branch in ((scratch.a, "task-a"), (scratch.b, "task-b")):
        run(main, "git", "worktree", "add", "-q", "-b", branch, str(tree))
    run(scratch.a, sys.executable, str(AGENT_WS), "label", "label of A")
    run(scratch.b, sys.executable, str(AGENT_WS), "label", "label of B")
    return scratch


@pytest.mark.parametrize("mode", MODES)
def test_worktree_commits_a_file_nobody_else_changed(
    scratch: Scratch, mode: str
) -> None:
    scratch.edit(scratch.b, "b.txt")  # B is busy elsewhere
    result = scratch.commit(scratch.a, "a.txt", mode)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_worktree_commit_of_a_file_another_worktree_changed_names_that_worktree(
    scratch: Scratch, mode: str
) -> None:
    assert scratch.commit(scratch.a, "a.txt", mode).returncode == 0
    scratch.edit(scratch.b, "a.txt")
    result = scratch.commit(scratch.a, "a.txt", mode)
    assert result.returncode != 0
    # B and only B, under B's own label — not the committing tree's.
    assert "a.txt is claimed by worktree wt-b [task-b] — label of B. " in result.stderr
    assert "label of A" not in result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_main_commits_a_file_no_worktree_changed(scratch: Scratch, mode: str) -> None:
    scratch.edit(scratch.a, "a.txt")
    scratch.edit(scratch.b, "b.txt")
    result = scratch.commit(scratch.main, "c.txt", mode)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", MODES)
def test_main_commit_of_a_file_a_worktree_changed_names_that_worktree(
    scratch: Scratch, mode: str
) -> None:
    scratch.edit(scratch.a, "a.txt")
    scratch.edit(scratch.b, "b.txt")
    result = scratch.commit(scratch.main, "b.txt", mode)
    assert result.returncode != 0
    assert "b.txt is claimed by worktree wt-b [task-b] — label of B. " in result.stderr
