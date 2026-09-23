"""Per-directory pytest config for ``lucid/test/unit/models``.

Pending families declare an explicit module-level skip in their test file.
Directory collect_ignore is deliberately not used: pytest bypasses it for
explicit file arguments, which the memory-bounded audit runner supplies.
Explicit skips remain visible and activate when the family is implemented.
"""
