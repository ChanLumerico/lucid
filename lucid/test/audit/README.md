# `lucid-audit` — the whole diagnosis, one command

Two stages, one verdict, one exit code.

| stage | asks | catches |
|---|---|---|
| **sweep** | does each reachable symbol keep its contract? | 1,507 symbols × 27 axes = 10,841 cells |
| **suite** | are the specific values the right values? | `pytest lucid/test` + a line-coverage floor |

They fail independently, and neither is a substitute for the other.
Measured on one session's defects: the sweep found the gradient that was
never wired and the sampler drawing at the wrong concentration; the
suite found the assignment writing a rectangle instead of a diagonal,
the histogram binning onto the wrong grid, the transform whose inverse
was NaN, and the window that deleted its own frame. A gate that runs
only one of them reports "clean" over half a framework.

```bash
lucid-audit                 # both stages, one verdict     (~15 min)
lucid-audit --audit-only    # the sweep alone
lucid-audit --tests-only    # the suite and the floor alone
```

```
── verdict ──────────────────────────────────────────────
  audit defects                   0
  audit coverage regressions      0
  suite failures                  0
  line coverage regressions       0

  clean on every stage that ran
```

Exit is `0` only when every stage that ran is clean, so this works as a
gate without reading the output. `2` means the harness itself broke —
distinct from `1`, which means the framework did.

### Why the suite runs in a subprocess

By the time the sweep is done the audit has imported the whole package
and, on some axes, patched parts of it. Collecting the suite into that
interpreter would let one stage's state decide the other stage's result,
which is the one thing a gate made of two independent checks must not
allow.

### Why line coverage is measured by default

Because it is free. The suite takes 9m09s uninstrumented and 8m29s under
`coverage` — the wall clock is MLX and Accelerate, not Python line
tracing. A floor that costs nothing to check should be checked on every
run rather than remembered. `--no-line-coverage` turns it off anyway.

The floor lives in `suite.json` and is compared per module, with a
2-point tolerance and a 20-statement minimum: adding statements to a
well-covered file lowers its percentage honestly, and a gate that fires
on that is a gate people learn to ignore. Record a new floor
deliberately:

```bash
lucid-audit --tests-only --update-suite
```

---

## Setup on a fresh machine

From the **repository root** (the directory holding `pyproject.toml`):

```bash
cd /path/to/lucid

# 1. Build the engine and install the package.
#    ⚠️ `uv pip`, never plain `pip` — .venv has no pip, so `pip` leaks to
#    the system pip3 and bakes that interpreter's absolute rpath into the
#    .so. See [[debug-build-wrong-venv-rpath]].
VIRTUAL_ENV=.venv MACOSX_DEPLOYMENT_TARGET=26.0 \
    uv pip install -e ".[audit]" --no-build-isolation

# 2. Confirm the engine linked against *this* venv.
otool -l lucid/_C/engine.cpython-314-darwin.so | grep -A2 LC_RPATH | grep path

# 3. Confirm the tool is on PATH and the surface enumerates.
lucid-audit --coverage
```

`--coverage` runs no probes at all — it walks the package and prints what
the sweep *can* reach. It is the fastest way to tell a working install
from a broken one.

### What the `[audit]` extra adds

`rich`, `coverage` and `pytest`. `rich` buys the live per-subsystem
display and nothing else — without it the sweep runs identically on a
stdlib ANSI console, because a correctness tool should not fail to start
over a presentation dependency. `coverage` backs the line-coverage floor
and `pytest` runs the suite stage; without either, that stage degrades
rather than fails: it says what is missing and the sweep still reports.

### If `lucid-audit` is not found

The console script is registered at install time, so it appears only
after step 1. Until then — and always, as a fallback — this is the same
program:

```bash
python -m lucid.test.audit --coverage
```

### Requirements

Inherited from Lucid itself: macOS 26+ on Apple Silicon, Python 3.14,
MLX ≥ 0.31. There is nothing the audit needs beyond what building the
engine already needs.

---

## Running it

```bash
lucid-audit                      # everything; minutes
lucid-audit --quick              # fewer domains, smaller probes
lucid-audit --json report.json   # machine-readable, with every SKIP reason
lucid-audit --fail-fast          # stop at the first defect
```

Scoping, when you are working on one area:

```bash
lucid-audit --subsystem nn.functional
lucid-audit --subsystem linalg,fft --axis grad,grad2
lucid-audit --select 'conv|pool'
```

Exit status is `0` when no defect survived, `1` when one did, `2` when
the harness itself broke.

### Through pytest

Every cell is also a collectable pytest case, deselected by default
because the sweep takes minutes:

```bash
pytest lucid/test/ -m audit
pytest lucid/test/audit/test_audit_smoke.py    # the harness's own tests, ~1s
```

---

## Reading the output

```
⠙ overall ━━━━━╺━━━━━━━━━━━━━━━━━  230/10841  16% 0:00:12 eta 0:04:31

  lucid         ━━━━━━╺━━━━━━━━━━ 230/807    221 ok     9 –
  special       ━━━━━━━━━━╺━━━━━━  53/114     47 ok    6 fail
```

One bar per subsystem, each with its own running counts. Defects print
above the live region as they are found.

### Statuses

| | |
|---|---|
| `PASS` | the axis's question was asked and answered correctly |
| `FAIL` | a defect |
| `TRNC` | the finite-difference probe was the limit, not the op — the disagreement shrank like `h²` when the step was refined |
| `GAUG` | correct only up to a symmetry (sign of an eigenvector, phase of an FFT) |
| `VAC` | the check ran but could not have failed — reported separately because a vacuous pass reads as coverage and is not |
| `UNSP` | the op refused, loudly and by design |
| `SKIP` | **the harness could not build inputs.** Not "no problem" — "not checked" |
| `KNWN` | a defect accepted in `known.json` |

`SKIP` is the number to watch. The summary's *applicable cells* line
reports how many cells produced a verdict at all, and that is the honest
measure of what a run actually checked.

---

## The three coverage numbers

They fail independently and are routinely conflated:

| | meaning |
|---|---|
| **reach** | fraction of public objects the surface enumerates. Checked against `independent_walk()`, which shares none of the enumeration's logic. Currently 100%. |
| **depth** | fraction of symbols with an axis that can actually fail, rather than only the smoke axis. Currently 93.6%; the remainder mutate process state and no numeric axis can express them. |
| **verdict** | fraction of cells that answer rather than SKIP. **Only a real run knows this**, and it is well below the other two. |

Reaching a symbol is not verifying it, and `depth` is an upper bound, not
a result. Quote the verdict rate from `--json` when you want to say how
much of the framework a run actually checked.

---

## Did my change break something?

An absolute count cannot answer that. An op that stops being reachable
moves one cell from `pass` to `unsupported` among fifteen hundred
already-unsupported ones, and nothing in the summary changes visibly.

So every run is diffed against `coverage.json`, which records which
`(axis, symbol)` cells produce a verdict:

```bash
lucid-audit                 # runs, then reports what moved
```

```
── coverage regressions · 1 ──────────────────────────────
  LOST  grad::lucid.expm1  (was pass, now unanswered)
```

The exit code is non-zero for a regression as well as for a defect, so
this works as a gate without reading the output. A cell that used to
answer and no longer does is the interesting direction: the op is still
exported, the audit simply cannot reach it any more — which is what a
refactor breaks without breaking a test.

Newly answered cells are reported too, as progress rather than failure.
When the new state is the intended one:

```bash
lucid-audit --update-coverage
```

Do that deliberately. Re-recording to make a red run go green silences
exactly the signal the file exists for.

Only verdicts are recorded. `SKIP` and `UNSUPPORTED` carry probe details
that move for reasons unrelated to the framework, and a baseline that
churns is a baseline nobody re-reads.

`--no-coverage-diff` turns the comparison off.

> `coverage.json` answers "is this still reachable". `known.json` answers
> "is this failure accepted". A failing cell belongs in the second; a cell
> that stopped being asked belongs in neither — fix it or record it as the
> new floor on purpose.

---

## Extending it

**A symbol is skipped.** Read the reason in the report. If the block is a
required parameter the derivation has no value for, add one entry to
`_autospec._BY_NAME` — that closes it for every other op spelling the
parameter the same way. Writing a per-op spec in `_specs.py` is the last
resort, not the first.

**A whole subsystem is thin.** `--coverage` prints per-subsystem depth. A
low number there means the symbols need an axis, not a spec — see
`_axes_data.py` for the shape of one.

**A finding looks wrong.** Check the harness before the framework. Of the
first twelve failures the newest axes reported, twelve were the
instrument: an abstract base raising `NotImplementedError` by contract, a
word-level tokenizer handed a vocabulary of single letters, a regex
parameter filled with the literal `"constant"`, a scheduler resumed
without restoring the optimizer it drives.

---

## Layout

| file | |
|---|---|
| `__main__.py` | CLI, the run loop, the summary |
| `_surface.py` | what there is to audit; the recursive walk and its independent check |
| `_axes.py` | the numeric axes and the `Axis` contract |
| `_axes_stability.py` | scale sweeps, extreme values, strides, determinism |
| `_axes_subsystem.py` | distributions, diffeq, quantization, serialization, compile, smoke |
| `_axes_data.py` | transforms, data, tokenizers, schedulers |
| `_specs.py` | hand-written invocations, and the tier order |
| `_autospec.py` | invocations derived from signatures |
| `_probe.py` | domains, sampling, finite differences |
| `_console.py` | the stdlib ANSI display |
| `_console_rich.py` | the `rich` display, when installed |
| `_result.py` | `Status`, `Finding`, `Report`, `Baseline` |
| `known.json` | accepted defects, reported as `KNWN` rather than `FAIL` |
