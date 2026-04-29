#!/usr/bin/env python3
"""Phase 6 acceptance criterion check — registry vs engine surface coverage.

The engine surface is intentionally broader than the op registry:

  Registry (OpRegistry::all())
    Tracks autograd-compute ops only — those that go through a CRTP kernel
    (UnaryKernel / BinaryKernel / ReduceKernel / NaryKernel) and have an
    OpSchema declared via LUCID_REGISTER_OP.

  Engine surface (dir(lucid._C.engine) + dir(lucid._C.engine.nn) + ...)
    Everything callable: ops, infrastructure, AMP API, constructors,
    inplace variants, nn sub-module, etc.

Acceptance criteria:
  1. Every registered autograd op is reachable somewhere in the engine
     (directly or via a sub-module like E.nn.*).
  2. The percentage of engine callables covered by the registry is tracked
     (target: ≥ 50% of simple compute ops in registry).
  3. No registered op is missing its schema name from the engine surface.

Usage:
    python tools/check_registry_coverage.py          # report
    python tools/check_registry_coverage.py --strict  # fail if any gaps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lucid._C import engine as E  # noqa: E402


def _all_engine_names() -> set[str]:
    """Collect all callable names reachable from the engine module."""
    names: set[str] = set()

    # Top-level
    for name in dir(E):
        if not name.startswith("_") and callable(getattr(E, name)):
            names.add(name)

    # Sub-modules (nn, linalg, etc.)
    for sub_name in dir(E):
        if sub_name.startswith("_"):
            continue
        sub = getattr(E, sub_name)
        if hasattr(sub, "__module__") and not callable(sub):
            continue
        try:
            for attr in dir(sub):
                if not attr.startswith("_") and callable(getattr(sub, attr)):
                    names.add(f"{sub_name}.{attr}")
        except Exception:
            pass

    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="Return non-zero if any registered op is unreachable.")
    args = parser.parse_args()

    all_schemas = list(E.op_registry_all())
    # Exclude internal backward-only nodes from coverage checks.
    schemas = [s for s in all_schemas if not s.internal]
    internal_schemas = [s for s in all_schemas if s.internal]
    engine_names = _all_engine_names()
    registered_names = {s.name for s in schemas}

    # --- Check 1: every non-internal registered op is reachable somewhere ---
    unreachable = []
    for s in schemas:
        # Direct hit
        if s.name in engine_names:
            continue
        # Sub-module hit (e.g. nn.dropout)
        sub_hit = any(n.endswith(f".{s.name}") for n in engine_names)
        if sub_hit:
            continue
        unreachable.append(s.name)

    # --- Check 2: engine callables that ARE in the registry ---
    top_level = {n for n in engine_names if "." not in n}
    top_in_registry = top_level & registered_names
    coverage_pct = 100.0 * len(top_in_registry) / max(len(top_level), 1)

    # --- Report ---
    print(f"Registry ops      : {len(schemas)} public + {len(internal_schemas)} internal")
    print(f"Engine callables  : {len(top_level)} top-level + sub-modules")
    print(f"Top-level in registry: {len(top_in_registry)} / {len(top_level)} "
          f"({coverage_pct:.0f}%)")

    if unreachable:
        print(f"\n⚠  {len(unreachable)} registered ops NOT reachable in engine:")
        for name in sorted(unreachable):
            print(f"   - {name}")
    else:
        print("\n✓ All registered ops are reachable in the engine surface.")

    # --- Classify why some top-level engine names are NOT in registry ---
    engine_only = top_level - registered_names
    infra = {n for n in engine_only if any(n.startswith(pfx) for pfx in
             ["amp_", "set_", "is_", "op_", "schema_", "engine_",
              "no_grad", "grad_mode", "autocast"])}
    constructors = {n for n in engine_only if n in
                    {"zeros","ones","full","empty","eye","arange","linspace",
                     "diag","zeros_like","ones_like","empty_like","full_like",
                     "rand","randn","randint","uniform","bernoulli"}}
    inplace = {n for n in engine_only if n.endswith("_")}
    remaining = engine_only - infra - constructors - inplace

    print(f"\nEngine-only breakdown ({len(engine_only)} total):")
    print(f"  Infrastructure (AMP/grad/registry API) : {len(infra)}")
    print(f"  Constructors (zeros/ones/rand/...)      : {len(constructors)}")
    print(f"  Inplace variants (*_)                  : {len(inplace)}")
    print(f"  Other compute (linalg/utils/einops/...) : {len(remaining)}")
    if remaining:
        print(f"  → {sorted(remaining)[:20]}")

    if unreachable and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
