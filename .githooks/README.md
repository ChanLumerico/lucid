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

| Hook          | Stage                | Behavior |
|---------------|----------------------|----------|
| `pre-commit`  | before commit drafted | Updates the Lines-of-Code badge in `README.md` via `cloc`, re-stages it.  Skip with `LUCID_SKIP_LOC_BADGE=1`. |
| `commit-msg`  | before commit lands   | Defensive warning if the staged diff doesn't include `CHANGELOG.md` and the commit subject uses a user-facing conventional-commit prefix (`feat:` / `fix:` / `perf:` / `refactor:` / `revert:` / `remove:` / `deprec:` / `sec:`). |
| `post-commit` | after commit lands    | **Auto-injects + amends `CHANGELOG.md`** for every user-facing conventional commit.  Calls `tools/changelog.py propose --auto --message-file .git/COMMIT_EDITMSG`, then `git commit --amend --no-edit --no-verify` to fold the entry into the same commit (preserving subject / body / author / date). A guard env var prevents the resulting amend from recursing. |

### Why post-commit and not prepare-commit-msg?

Git's commit pipeline finalises the *tree* of the commit at the end of
the `pre-commit` hook — any `git add` performed in
`prepare-commit-msg` or `commit-msg` modifies the index for the *next*
commit, not the current one.  `post-commit` runs after the commit
hash has been computed but before push, so amending in place is safe.

The post-commit auto-amend keeps `[Unreleased]` **in sync with the
conventional-commit history with zero human steps**.  Manual
``tools/changelog.py add`` is reserved for entries that don't map
1-to-1 to a commit subject (e.g. a multi-commit feature grouped under
one bullet).

`commit-msg` is still **advisory** — it never aborts a commit.  Strict
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
