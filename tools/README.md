# tools/

Developer and CI tooling for Lucid. Nothing here is imported by `lucid/`.
Most scripts carry their own usage in their module docstring; this file
documents the ones whose rules other people's work depends on.

## Quality gate

`tools/quality_gate/` plus the committed `tools/quality_baseline.json`: a
**ratchet** on measured code-health debt. Existing debt never blocks a
commit, and new debt is zero.

### What it counts

Each tool's findings become `(tool, rule, file) -> count`. Production
Python is `lucid/` minus `lucid/test/`; C++ is `lucid/_C/`.

| collector | measures | defect class |
|---|---|---|
| `ruff` | the project's lint config plus `B BLE S110 TRY PGH C90 PLR0911/0912/0915 ERA ARG SIM RET PIE PERF UP RUF100` (`TRY003` off) | D1, D3, D4, D7, style |
| `counters` | `type: ignore[code]` per code (a bare ignore is `type-ignore-bare`), `cast(`, `noqa[CODE]`, lines over the file cap (1500 py / 3000 C++) | D1, D3 |
| `lizard` | functions over 100 lines or CCN 15 (py + C++); `_`-internal Python functions over 7 parameters | D3 |
| `vulture` | dead code at 60 % confidence; scans tests as users of the API, counts only production files | D7 |
| `jscpd` | clone pairs of 8+ lines / 50+ tokens (py + C++), via `npx jscpd@5.4.0` | D2 |
| `mypy` | `mypy.ini` (strict) plus `warn-unused-ignores`, `ignore-without-code`, `redundant-cast`, `redundant-expr`, `truthy-bool`, `possibly-undefined` | D1, D4 |
| `semgrep` | Lucid rules in `tools/semgrep/rules/` (none until DS-4) | D5 |
| `clang-tidy` | stub, off by default (DS-3) | — |

Deliberately off: `FBT` and public-API `PLR0913` (signatures follow the
reference framework's), `TRY003` (a library's exceptions carry their own
messages). The thresholds live in `tools/quality_gate/config.py` and are
recorded in the baseline. The tool versions are pinned in the `quality`
dependency group in `pyproject.toml` (`pip install --group quality`),
because a different version counts differently.

### Modes

```
python -m tools.quality_gate --fast        # pre-commit: files changed since HEAD; ruff, counters, lizard + jscpd (~3 s)
python -m tools.quality_gate --full        # CI: whole tree, every collector (~40 s cold on an M1 Pro)
python -m tools.quality_gate --diff main   # SLOP DELTA for a worker report and for land
python -m tools.quality_gate --report      # totals per defect class (deslop SWEEP)
python -m tools.quality_gate --update      # record decreases: lowers the baseline, never raises
python -m tools.quality_gate --rebaseline --reason "<why>"   # orchestrator only: may raise
```

`--collectors ruff,counters` narrows any mode. Exit codes: `0` passes,
`1` means a count rose (or the baseline is out of date), and `2` means the
gate could not measure. A tool that crashes is never read as "no findings".

### The rules it enforces

- **No count may rise.** A new `type: ignore`, a new `except Exception: pass`,
  a 200-line function or a pasted block each fails the gate.
- **A count that fell must be recorded in the same commit.** `check` fails
  on a baseline that is higher than the code, because that slack would let
  slop come back unnoticed. Run `--update` (or `--update --fast` after a
  fast check) and stage `tools/quality_baseline.json`. CI passes
  `--allow-slack`, so slack there is only a warning. The fast path cannot
  see a fall in a cross-file count (mypy, vulture), and two lands can
  combine into one. The orchestrator clears such slack with `--update` on
  main after a land. Nothing can slip back in through it in the meantime:
  `land` measures the counts on main itself, not main's baseline.
- **Moving a file is not new debt.** Renames since the commit that last
  wrote the baseline (git's `-M`) carry their counts to the new path, and
  `--update` rewrites the keys. A move combined with a heavy rewrite falls
  below git's similarity threshold and reads as delete + add. Record a move
  in its own commit when that matters.
- **Counts, not sites.** Moving a finding within a file is free. Swapping
  one finding for another of the same rule in the same file is not
  detected. That is the price of not churning on every edit.

### Where it runs

| layer | command | blocks |
|---|---|---|
| pre-commit (`.githooks/pre-commit` §1b, `.pre-commit-config.yaml`) | `--fast` when `lucid/` or the baseline is staged | the commit (`LUCID_SKIP_QUALITY_GATE=1` skips; CI and land still check) |
| CI (`ci.yml` job `quality-gate`) | `--full --allow-slack` | the push |
| `agent_ws.py land` | `--diff main` on the rebased branch; `land --check` diffs against the fork point | the land (exit 6). `--allow-slop "<reason>"` overrides, and every use is appended to `~/Library/Caches/lucid-agents/slop-overrides.log` |

`--diff REF` materialises REF's tree with `git archive` (no worktree, so the
claim gate never sees it). It measures both trees: the local collectors on
the files that differ, the cross-file ones on everything. It then also
checks the working tree against the committed baseline. On a land, that
means the baseline the rebase merged.

### Parallel branches and the baseline

Every branch that lowers a count writes the baseline, so the file is
claim-exempt (`agent_ws.GENERATED`). It merges through the
`lucid-quality-baseline` git merge driver (`.gitattributes`). The driver
does a per-key 3-way merge:

- a key that one side left alone takes the other side's value;
- a key that both sides lowered takes the lower one;
- a key deleted on either side stays deleted.

Each side can only lower counts, so the result can never be higher than the
real combined state. A side that **raises** a key aborts the merge, unless
that side also carries a new `--rebaseline` record. `agent_ws.py setup`
registers the driver in repo-local config, and `land` re-checks it before
every rebase. Without the driver, git falls back to a text merge, which
conflicts on neighbouring keys.

### Unlaundering (redundant casts, unused ignores)

When an owner's types are fixed (LCD-260: `Module.__call__` typed from the
subclass's `forward`), the casts and `type: ignore`s written around the
old types become dead, and mypy can prove which ones. Those are removed
mechanically, never by hand:

```
python -m tools.quality_gate.unlaunder --paths lucid/models/vision          # edit
python -m tools.quality_gate.unlaunder --paths lucid/models/vision --check  # count, exit 1 if any
```

Each round runs mypy over `lucid/` with the gate's flags. It then unwraps
`cast(T, e)` to `e` at each `redundant-cast` under `--paths`, and drops
the codes each `unused-ignore` names. An ignore left with no codes goes
together with its trailing reason; a `# noqa` or `# pragma` tail stays.
mypy reports one finding per line and message, so the next cast on a line
only shows up after the first one is gone. Rounds therefore repeat until
nothing more applies, usually in 2–3 rounds. A round takes about 6 s
once mypy's cache (`.mypy_cache/unlaunder`) is warm, and about 25 s cold.
The heavy lock is not needed. The
touched files then get `ruff --select F401 --fix` (for a `cast` import
that became unused) and black at 88. Rules that keep the edit safe:

- every file's result is checked against the original AST with the casts
  replaced, and parentheses are added when the value binds looser than
  its new context;
- a cast whose removed parts hold a comment is left in place and listed;
- a mypy error of any other code that appears during the run fails it
  (exit 2);
- a `--paths` entry outside `lucid/` (or under `lucid/test/`) is
  refused, because mypy would never report there and the result would
  read as "0 left".

Exit codes are `0` (nothing left), `1` (sites left) and `2` (could not
proceed).

**The transitional-section convention.** A change to an owner's types
turns hundreds of existing suppressions into errors at once under the
gate's `--warn-unused-ignores` / `redundant-cast`. Those suppressions
cannot all be removed in the same commit. The owner's commit therefore
adds a `mypy.ini` section for the affected packages that turns those two
checks off, and the comment right above its header starts with
`# transitional (<issue>)`. unlaunder runs mypy on a copy of `mypy.ini`
without every section introduced that way, so it sees exactly what the
section hides. Only the comment block directly above a header counts: a
key line between the marker and the header cancels it. The last sweep
card deletes the section (LCD-260-C4).

**Per sweep card** (one path set, for example `lucid/models/vision`):

1. `python -m tools.quality_gate.unlaunder --paths <prefix>...`
2. `python -m tools.quality_gate.unlaunder --paths <prefix>... --check` → `0 redundant-cast, 0 unused-ignore`.
3. `mypy --strict lucid/` → 0 errors (with the real `mypy.ini`).
4. `python -m tools.quality_gate --update --fast` to record the drop. Stage
   `tools/quality_baseline.json` together with the edits.

The edits change no behaviour, so the card's tests are the touched
package's own suite.

### Adding a collector

1. Put a subclass of `collectors.Collector` in `tools/quality_gate/collectors/`.
   Set `local` if a file's count depends on that file alone, `fast` if it
   is cheap enough for pre-commit, and `default`/`in_diff`. Register it in
   `all_collectors()`.
2. Map its rules to a defect class in `config._CLASS_RULES`.
3. Its existing findings raise counts, so the orchestrator runs
   `--rebaseline --reason "<collector> added"` in the same commit.
4. Seed a regression for it in `tools/test_quality_gate.py`.

Tests: `.venv/bin/python3 -m pytest tools/test_quality_gate.py` (scratch git
repositories; the jscpd case skips without Node).
