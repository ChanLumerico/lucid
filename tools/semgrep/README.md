# Lucid semgrep rules

Lucid-specific structural rules that ruff, mypy and lizard cannot express —
"an index-range rejection goes through the shared validator", "no raw MLX
gather outside the engine boundary".  They are counted by the quality gate's
`semgrep` collector (`tools/quality_gate/collectors/semgrep_rules.py`) like
any other finding: existing hits sit in the baseline, a new one fails.

The rules themselves arrive with DS-4 (LCD-263, error taxonomy).  Until a
file exists in `rules/`, the collector measures nothing.

## Adding a rule

1. One YAML file per defect class in `rules/` (`*.yml` / `*.yaml`), each rule
   with an `id` (the gate's rule name is the id's last dot-component), a
   `message` that says what to call instead, and `languages: [python]` or
   `[cpp]`.
2. Test it on the tree first:
   `.venv/bin/semgrep scan --config tools/semgrep/rules/<file>.yml lucid/`.
3. Record the hits that already exist — that raises counts, so it is the
   orchestrator's `python -m tools.quality_gate --rebaseline --reason "semgrep: <rule>"`
   in the same commit as the rule.

Only the OSS CLI is used (`--metrics=off`); no registry rules are pulled.
