"""What the gate measures, and why each threshold sits where it does.

Every number here is part of the ratchet's contract: changing one changes
what every count in ``tools/quality_baseline.json`` means, so a change here
ships with ``--rebaseline --reason`` in the same commit.
"""

BASELINE_PATH = "tools/quality_baseline.json"
BASELINE_VERSION = 1

# ── Scope ─────────────────────────────────────────────────────────────────
# Production Python and the C++ engine.  Tests are excluded from the counts
# (they are scanned by vulture as *users* of the API, see deadcode.py).
PY_ROOT = "lucid/"
PY_EXCLUDE_PREFIXES = ("lucid/test/",)
CPP_ROOT = "lucid/_C/"
CPP_SUFFIXES = (".cpp", ".h", ".hpp", ".mm", ".cc")

# ── ruff ──────────────────────────────────────────────────────────────────
# Added on top of the project's own [tool.ruff.lint] (E, F, W) with
# --extend-select, so RUF100 judges a `noqa` against the rules the project
# actually enables.  Deliberately off:
#   FBT      — reference-compatible signatures take positional bools.
#   TRY003   — a library's exceptions carry their own messages.
#   PLR0913  — public signatures mirror the reference framework's; argument
#              counts are judged by lizard on `_` internals only.
RUFF_EXTEND_SELECT = (
    "B,BLE,S110,TRY,PGH,C90,PLR0911,PLR0912,PLR0915,ERA,ARG,SIM,RET,PIE,PERF,UP,RUF100"
)
RUFF_IGNORE = "TRY003"

# ── lizard (Python and C++) ───────────────────────────────────────────────
# Physical length in lines.  100 keeps the long tail visible (260 functions
# on 2026-10-05; 83 over 150) while a new function's budget stays ~80.
FUNCTION_LENGTH_MAX = 100
# Cyclomatic complexity: lizard's own default warning level.
FUNCTION_CCN_MAX = 15
# Parameters, for `_`-prefixed Python internals only — public signatures
# follow the reference framework and are not ours to shorten.
INTERNAL_ARGS_MAX = 7

# ── file size cap ─────────────────────────────────────────────────────────
# The count is the number of lines *over* the cap, so a file already over it
# may not grow and a new file may not start over it.
LOC_CAP_PY = 1500
LOC_CAP_CPP = 3000
# Rules whose count is a size, not a number of findings: class totals count
# one per file so 14954-line headers do not drown every other class.
PER_FILE_RULES = frozenset({"file-loc-over-cap"})

# ── vulture ───────────────────────────────────────────────────────────────
VULTURE_MIN_CONFIDENCE = 60
VULTURE_IGNORE_NAMES = ("__*__",)
VULTURE_WHITELIST = "tools/quality_gate/vulture_whitelist.py"

# ── jscpd ─────────────────────────────────────────────────────────────────
# 8 lines / 50 tokens: the 2026-10-05 probe's "8-line window repeated in 3+
# files" class.  Pinned: a different jscpd tokenises differently.
JSCPD_VERSION = "5.4.0"
JSCPD_MIN_LINES = 8
JSCPD_MIN_TOKENS = 50
JSCPD_FORMATS = "python,cpp,objectivec"

# ── mypy (full mode only — the slow one) ──────────────────────────────────
# On top of mypy.ini (strict).  `--platform darwin` so a Linux CI runner
# sees the same `sys.platform` branches as the Macs.
MYPY_FLAGS = (
    "--warn-unused-ignores",
    "--enable-error-code=ignore-without-code",
    "--enable-error-code=redundant-cast",
    "--enable-error-code=redundant-expr",
    "--enable-error-code=truthy-bool",
    "--enable-error-code=possibly-undefined",
    "--platform=darwin",
)

# ── semgrep ───────────────────────────────────────────────────────────────
SEMGREP_RULES_DIR = "tools/semgrep/rules"

# ── defect classes (obsidian roadmap-deslop-quality-gate) ─────────────────
CLASSES = {
    "D1": "type laundering",
    "D2": "duplication",
    "D3": "god units",
    "D4": "silent errors",
    "D5": "error taxonomy",
    "D7": "dead code",
    "S": "style / modernise",
}

# (tool, rule prefix) -> class; the longest matching prefix wins.
_CLASS_RULES = (
    ("counters", "type-ignore", "D1"),
    ("counters", "cast", "D1"),
    ("counters", "noqa", "D1"),
    ("counters", "file-loc", "D3"),
    ("ruff", "PGH", "D1"),
    ("ruff", "C90", "D3"),
    ("ruff", "PLR", "D3"),
    ("ruff", "B", "D4"),
    ("ruff", "BLE", "D4"),
    ("ruff", "S110", "D4"),
    ("ruff", "TRY", "D4"),
    ("ruff", "ARG", "D7"),
    ("ruff", "ERA", "D7"),
    ("ruff", "RUF100", "D7"),
    ("ruff", "", "S"),
    ("mypy", "unused-ignore", "D1"),
    ("mypy", "ignore-without-code", "D1"),
    ("mypy", "redundant-cast", "D1"),
    ("mypy", "", "D4"),
    ("lizard", "", "D3"),
    ("vulture", "", "D7"),
    ("jscpd", "", "D2"),
    ("semgrep", "", "D5"),
    ("clang-tidy", "", "D4"),
)


def class_of(tool: str, rule: str) -> str:
    best = ("", "S")
    for t, prefix, cls in _CLASS_RULES:
        if t == tool and rule.startswith(prefix) and len(prefix) >= len(best[0]):
            best = (prefix, cls)
    return best[1]


def thresholds() -> dict[str, object]:
    """Recorded in the baseline so a silent config change is visible."""
    return {
        "function_length_max": FUNCTION_LENGTH_MAX,
        "function_ccn_max": FUNCTION_CCN_MAX,
        "internal_args_max": INTERNAL_ARGS_MAX,
        "loc_cap_py": LOC_CAP_PY,
        "loc_cap_cpp": LOC_CAP_CPP,
        "ruff_extend_select": RUFF_EXTEND_SELECT,
        "ruff_ignore": RUFF_IGNORE,
        "vulture_min_confidence": VULTURE_MIN_CONFIDENCE,
        "jscpd": f"{JSCPD_VERSION} lines>={JSCPD_MIN_LINES} tokens>={JSCPD_MIN_TOKENS}",
        "mypy_flags": " ".join(MYPY_FLAGS),
    }
