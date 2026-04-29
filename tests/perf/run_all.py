#!/usr/bin/env python3
"""
Run all Lucid performance benchmarks and print a markdown report.

Usage (from tests/perf/):
  python run_all.py             # plain text output
  python run_all.py --md        # markdown table
  python run_all.py --n 20      # fewer iterations (faster)

Output written to stdout; redirect to docs/perf/RESULTS.md if desired.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make _runner importable
sys.path.insert(0, str(Path(__file__).parent))

from _runner import BenchSpec, GPU_OK, MPS_OK, run_suite  # noqa: E402

import bench_matmul
import bench_conv2d
import bench_softmax_sdpa
import bench_norm


SUITES: list[tuple[str, list[BenchSpec]]] = [
    ("Matmul",          bench_matmul.SPECS),
    ("Conv2d",          bench_conv2d.SPECS),
    ("Softmax + SDPA",  bench_softmax_sdpa.SPECS),
    ("Norm",            bench_norm.SPECS),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", action="store_true", help="Emit markdown tables")
    parser.add_argument("--n", type=int, default=50, help="Iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=8, help="Warm-up iterations")
    parser.add_argument("--suite", default=None, help="Run only this suite (partial match)")
    parser.add_argument("--save", action="store_true",
                        help="Save markdown output to tests/perf/baseline.md")
    args = parser.parse_args()
    if args.save:
        args.md = True

    import contextlib, datetime, io

    def _run_body(file=sys.stdout):
        def p(*a, **kw): print(*a, **kw, file=file)
        p(f"# Lucid Performance Baseline")
        p(f"")
        p(f"**Date:** {datetime.date.today()}  ")
        p(f"**Backends:** lucid-cpu, {'lucid-gpu, ' if GPU_OK else ''}torch-cpu"
          f"{', torch-mps' if MPS_OK else ''}  ")
        p(f"**Iterations:** {args.n} (warm-up: {args.warmup})  ")
        p(f"**`% of torch`** = lucid-cpu latency ÷ torch-cpu latency × 100 "
          f"(100% = same speed, <100% = faster)")
        p()
        with contextlib.redirect_stdout(file):
            for suite_name, specs in SUITES:
                if args.suite and args.suite.lower() not in suite_name.lower():
                    continue
                p(f"## {suite_name}" if args.md else f"=== {suite_name} ===")
                run_suite(specs, n_warmup=args.warmup, n_iter=args.n, markdown=args.md)
                p()

    if args.save:
        buf = io.StringIO()
        _run_body(file=buf)
        dest = Path(__file__).parent / "baseline.md"
        dest.write_text(buf.getvalue())
        print(f"Saved → {dest}")
    else:
        _run_body()


if __name__ == "__main__":
    main()
