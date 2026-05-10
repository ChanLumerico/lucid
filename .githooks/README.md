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

| Hook         | Stage              | Behavior |
|--------------|--------------------|----------|
| `commit-msg` | before commit lands | Warns (does **not** block) when the commit subject uses a user-facing conventional-commit prefix (`feat:` / `fix:` / `perf:` / `refactor:` / `revert:` / `remove:` / `deprec:` / `sec:`) but the staged diff doesn't include `CHANGELOG.md`. |
| `post-commit`| after commit lands  | Same detection logic — prints either a confirmation (`📝 CHANGELOG.md updated alongside this commit ✓`) or a follow-up nudge with the exact `tools/changelog.py` invocation to add the entry. |

Both hooks are **advisory** — they never abort a commit.  Strict
enforcement is delegated to CI via:

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
