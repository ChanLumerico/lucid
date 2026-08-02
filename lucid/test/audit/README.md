# The audit sweep

An exhaustive `(symbol × axis)` correctness sweep over every public Lucid
API outside the model zoo. 1,507 symbols, 27 axes, 10,841 cells.

It is not a substitute for `pytest lucid/test/` — it is the thing that
tells you what those tests are *not* asking.

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

`rich`, and only `rich`. It buys the live per-subsystem display; without
it the sweep runs identically on a stdlib ANSI console. A correctness
tool should not fail to start over a presentation dependency, so if you
are scripting this in CI, plain `-e .` is enough.

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
