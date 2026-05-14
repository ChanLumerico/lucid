# `.githooks/` — Lucid local git hooks

Project-tracked git hooks that keep `CHANGELOG.md` in sync with user-facing
commits.

## Enable for this clone

Run **once** per checkout:

```bash
git config core.hooksPath .githooks
```

This tells git to look for hook scripts under `.githooks/` (which is
git-tracked) instead of the default `.git/hooks/` (which is local-only and
not shareable).

To revert:

```bash
git config --unset core.hooksPath
```

## What's installed

| Hook                  | Stage                              | Behavior |
|-----------------------|------------------------------------|----------|
| `pre-commit`          | before commit msg drafted          | Updates the Lines-of-Code badge in `README.md` via `cloc`, re-stages it.  Skip with `LUCID_SKIP_LOC_BADGE=1`. |
| `prepare-commit-msg`  | after msg drafted, before commit lands | **Auto-injects a `CHANGELOG.md` entry** under `[Unreleased]` for every user-facing conventional commit (`feat:` / `fix:` / `perf:` / `refactor:` / `revert:` / `remove:` / `deprec:` / `sec:`).  Reads the staged commit message file directly (HEAD hasn't moved yet), calls `tools/changelog.py propose --auto --message-file <msg>`, and stages the updated `CHANGELOG.md` so it lands in the same commit. |
| `commit-msg`          | before commit lands                | Defensive warning if the staged diff *still* doesn't include `CHANGELOG.md` (e.g. when the commit isn't conventional, or when the user opted out). |
| `post-commit`         | after commit lands                 | Confirms (`📝 CHANGELOG.md updated alongside this commit ✓`) or nudges with the exact `tools/changelog.py` invocation. |

The `prepare-commit-msg` hook makes `[Unreleased]` **automatically
stay in sync** with the conventional-commit history — manual
``tools/changelog.py add`` is reserved for entries that don't map 1-to-1
to a commit subject (e.g. multi-commit features grouped under one bullet).

`commit-msg` / `post-commit` are still **advisory** — they never abort
a commit.  Strict enforcement is delegated to CI via:

```bash
python tools/changelog.py check --strict
```

## Bypass

For a single commit:

```bash
LUCID_SKIP_CHANGELOG_CHECK=1 git commit -m "..."
```

For all commits temporarily:

```bash
git config --unset core.hooksPath
```

`--no-verify` also works but is discouraged because it bypasses every hook
including pre-commit linters.

## Adding entries

```bash
python tools/changelog.py propose                       # interactive — reads HEAD
python tools/changelog.py add added   "Tensor.foo()"    # explicit
python tools/changelog.py add fixed   "edge case in X"
python tools/changelog.py add perf    "Y kernel 30% faster"
python tools/changelog.py add tooling "ruff config tweak"
```

Categories follow [Keep a Changelog](https://keepachangelog.com/) with two
project-specific extensions: `performance` and `tooling`.

Releasing:

```bash
python tools/changelog.py release 3.0.1
```

This promotes the current `[Unreleased]` content into a dated `[3.0.1] -
YYYY-MM-DD` section, resets `[Unreleased]` to a placeholder, and updates the
compare links at the bottom of the file.
