"""A broken instrument cannot turn a required CI stage green, and a stage
failure in ``part_checks`` / ``part_core`` / ``Coverage`` does not mask later
stages (LCD-530)."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "stage,next_marker",
    [
        ("Symbol x axis audit", "# Docstring examples"),
        (
            "Docstring examples (against the doctest floor)",
            "# Model-zoo family contract",
        ),
    ],
)
@pytest.mark.parametrize("status", [0, 1, 2])
def test_required_ci_stage_preserves_failure(
    tmp_path: Path,
    stage: str,
    next_marker: str,
    status: int,
) -> None:
    root = Path(__file__).resolve().parents[4]
    script = (root / "scripts/ci_full.sh").read_text()
    block = script.split(f'echo "==> {stage}"', 1)[1].split(next_marker, 1)[0]
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport os, sys\nsys.exit(int(os.environ['TEST_STAGE_STATUS']))\n"
    )
    fake_python.chmod(0o700)
    harness = (
        "set -euo pipefail\n"
        "failed_stages=()\n"
        + block
        + "\nprintf 'NEXT_STAGE\\n'\n"
        + 'if [ "${#failed_stages[@]}" -ne 0 ]; then exit 1; fi\n'
    )
    result = subprocess.run(
        ["bash", "-c", harness],
        env={
            **os.environ,
            "PYTHON_BIN": str(fake_python),
            "TEST_STAGE_STATUS": str(status),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stdout + result.stderr
    # Exit 1 records the failure and continues to later stages; exit 2
    # ("harness broke") stops immediately.
    assert ("NEXT_STAGE" in result.stdout) == (status in (0, 1))


def test_part_core_and_coverage_do_not_abort_on_first_failure_or_collection_error() -> None:
    root = Path(__file__).resolve().parents[4]
    ci_full = (root / "scripts/ci_full.sh").read_text()
    part_core = ci_full.split("part_core() {", 1)[1].split("\n}", 1)[0]
    assert " -x" not in part_core
    assert "--maxfail=50" in part_core
    assert "--continue-on-collection-errors" in part_core

    ci_yml = (root / ".github/workflows/ci.yml").read_text()
    cov_lines = [
        line.strip()
        for line in ci_yml.splitlines()
        if "coverage run --parallel-mode" in line
    ]
    assert len(cov_lines) == 3
    for line in cov_lines:
        assert "--continue-on-collection-errors" in line


def test_part_checks_stage_helper_continues_and_exits_one_at_end(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[4]
    ci_full = (root / "scripts/ci_full.sh").read_text()
    part_checks = ci_full.split("part_checks() {", 1)[1].split("\npart_zoo()", 1)[0]
    stage_def = part_checks.split("# ── 3. Parity tier", 1)[0]
    summary_tail = '    if [ "${#failed_stages[@]}" -ne 0 ]; then' + (
        part_checks.split('    if [ "${#failed_stages[@]}" -ne 0 ]; then', 1)[1].rsplit(
            "\n}", 1
        )[0]
    )
    summary_file = tmp_path / "step_summary.md"
    script = (
        "set -euo pipefail\n"
        "run_test() {\n"
        + stage_def
        + '\nstage "First failing stage" bash -c "exit 1"\n'
        + 'stage "Second passing stage" printf "RAN_SECOND\\n"\n'
        + summary_tail
        + "\n}\nrun_test\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary_file)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "RAN_SECOND" in result.stdout
    assert "First failing stage" in result.stderr
    assert "First failing stage" in summary_file.read_text()


def test_pyproject_test_extra_includes_quality_gate_collectors() -> None:
    """``[project.optional-dependencies].test`` must include lizard and vulture for unit tests."""
    import tomllib

    root = Path(__file__).resolve().parents[4]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    test_deps = data["project"]["optional-dependencies"]["test"]
    quality_deps = data["dependency-groups"]["quality"]
    for pkg in ("lizard", "vulture"):
        pinned = next(d for d in quality_deps if d.startswith(f"{pkg}=="))
        assert pinned in test_deps, f"{pinned!r} missing from [project.optional-dependencies].test"

