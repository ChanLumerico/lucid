"""Ratchet quality gate — measured code-health debt may only go down.

Every tool's findings become ``(tool, rule, file) -> count``; the committed
``tools/quality_baseline.json`` holds the current counts, and the gate fails
when any count rises.  Existing debt never blocks; new debt is zero.

    python -m tools.quality_gate --fast        # pre-commit: changed files
    python -m tools.quality_gate --full        # CI: everything
    python -m tools.quality_gate --diff main   # SLOP DELTA for a report / land
    python -m tools.quality_gate --report      # totals per defect class
    python -m tools.quality_gate --update      # record decreases (lower only)

Usage and design: tools/README.md "Quality gate".
"""
