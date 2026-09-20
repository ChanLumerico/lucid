"""Merge explicitly ordered checkpoint reports without hiding later failures.

Later inputs replace earlier observations of the same factory, including failed
or unverified observations. Each selected row retains its report provenance.
Completeness means recorded default-checkpoint comparisons, not identical source
trees, all weight variants, task metrics, or training equivalence.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def merge(paths: list[Path], expected: set[str]) -> dict[str, Any]:
    observations: dict[str, dict[str, object]] = {}
    sources: list[dict[str, object]] = []
    for path in paths:
        payload = path.read_bytes()
        report = json.loads(payload)
        if not isinstance(report, dict) or not isinstance(report.get("evidence"), list):
            raise ValueError(f"{path}: expected a checkpoint evidence report")
        rows = report["evidence"]
        if report.get("processed", len(rows)) != len(rows):
            raise ValueError(f"{path}: processed count does not match evidence")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("model"), str):
                raise ValueError(f"{path}: every observation needs a model name")
            name = row["model"]
            if name in seen:
                raise ValueError(f"{path}: duplicate observation for {name}")
            seen.add(name)
            observations[name] = {**row, "evidence_report": len(sources)}
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "environment": report.get("environment"),
                "version": report.get("version"),
                "finished": report.get("finished"),
                "complete": report.get("complete"),
                "scope": report.get("scope"),
                "problems": report.get("problems"),
                "unreachable": report.get("unreachable"),
            }
        )

    missing = sorted(expected - observations.keys())
    extra = sorted(observations.keys() - expected)
    evidence = []
    problems = []
    compared = unverified = 0
    for name in sorted(expected & observations.keys()):
        row = observations[name]
        evidence.append(row)
        if any(key in row for key in ("unreachable", "unsupported", "degenerate")):
            unverified += 1
            continue
        if "error" in row:
            problems.append(f"{name}: {row['error']}")
            continue
        difference, tolerance = row.get("max_diff"), row.get("tolerance", 1e-4)
        if (
            not isinstance(difference, (int, float))
            or isinstance(difference, bool)
            or not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool)
            or not math.isfinite(difference)
            or not math.isfinite(tolerance)
            or difference < 0
            or tolerance <= 0
            or not isinstance(row.get("top1_agrees"), bool)
        ):
            problems.append(f"{name}: invalid or incomplete numeric evidence")
            continue
        compared += 1
        if difference > tolerance or not row["top1_agrees"]:
            problems.append(f"{name}: numeric or ranking mismatch")
    return {
        "schema": 1,
        "selected": len(expected),
        "processed": len(evidence),
        "finished": not missing,
        "compared": compared,
        "unreachable": unverified,
        "problems": problems,
        "missing": missing,
        "unexpected": extra,
        "complete": bool(expected) and compared == len(expected) and not problems and not extra,
        "evidence": evidence,
        "reports": sources,
        "source_match": "historical reports; working-tree equivalence is not inferred",
        "selection": "explicit input order; later observations replace earlier ones, regardless of verdict",
        "scope": "recorded default-checkpoint output comparisons only; no dataset metric, all-variant, or training claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Populate the existing registry, without constructing models.
    import lucid.models  # noqa: F401
    from lucid.weights._registry import _WEIGHTS_BY_MODEL

    try:
        result = merge(args.reports, set(_WEIGHTS_BY_MODEL))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "selected",
                    "processed",
                    "compared",
                    "unreachable",
                    "missing",
                    "problems",
                    "complete",
                )
            }
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
