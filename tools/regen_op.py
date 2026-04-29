#!/usr/bin/env python3
"""Phase 6.5.4 — Boilerplate-only op regeneration.

Re-emits the schema declaration and op-function stub in an existing op .h
and .cpp WITHOUT overwriting any hand-written compute code (dispatch(),
cpu_kernel(), gpu_kernel(), grad_formula(), forward()).

Use this when:
  - An OpSchema field changes (version bump, AmpPolicy, determinism note).
  - The op name in the schema drifts from the expected name.
  - You want to sync a scaffolded op's boilerplate after renaming.

Usage:
    python tools/regen_op.py ufunc.cube_root
    python tools/regen_op.py ufunc.cube_root --amp-policy ForceFP32 --dry-run

The tool only touches:
  1. The `schema_v1` definition in the .cpp file.
  2. The `schema_v1` declaration comment in the .h file.
  It leaves ALL other code intact.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS_DIR = ROOT / "lucid" / "_C" / "ops"


def _snake_to_camel(s: str) -> str:
    return "".join(w.capitalize() for w in s.split("_"))


def _parse_amp(amp_arg: str) -> str:
    mapping = {
        "promote": "Promote",
        "keepinput": "KeepInput",
        "keep_input": "KeepInput",
        "forcefp32": "ForceFP32",
        "force_fp32": "ForceFP32",
    }
    return mapping.get(amp_arg.lower(), amp_arg)


def regen(family: str, snake: str, amp: str, det: bool, version: int,
          dry_run: bool) -> int:
    camel = _snake_to_camel(snake)
    cpp_path = OPS_DIR / family / f"{camel}.cpp"

    if not cpp_path.exists():
        print(f"[regen_op] ERROR: {cpp_path} not found. Run new_op.py first.")
        return 1

    det_str = "true" if det else "false"
    new_schema = (
        f'const OpSchema {camel}Backward::schema_v1{{\n'
        f'    "{snake}", {version}, AmpPolicy::{amp}, /*deterministic=*/{det_str}}};'
    )

    text = cpp_path.read_text()

    # Find and replace the schema_v1 definition
    pattern = re.compile(
        rf'const OpSchema {re.escape(camel)}Backward::schema_v1\{{[^;]+\}};',
        re.DOTALL
    )
    match = pattern.search(text)
    if not match:
        print(f"[regen_op] ERROR: could not find schema_v1 in {cpp_path}.")
        print("  Pattern expected:  const OpSchema {camel}Backward::schema_v1{{...}};")
        return 1

    old_schema = match.group(0)
    if old_schema == new_schema:
        print(f"[regen_op] {cpp_path.name}: schema already up-to-date, nothing to do.")
        return 0

    if dry_run:
        print(f"[regen_op] DRY RUN — would replace in {cpp_path.relative_to(ROOT)}:")
        print(f"  OLD: {old_schema!r}")
        print(f"  NEW: {new_schema!r}")
        return 0

    new_text = text[:match.start()] + new_schema + text[match.end():]
    cpp_path.write_text(new_text)
    print(f"[regen_op] updated schema_v1 in {cpp_path.relative_to(ROOT)}")
    print(f"  AmpPolicy={amp}, deterministic={det_str}, version={version}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("op_name",
                        help="Op in family.snake_name format (e.g. ufunc.cube_root)")
    parser.add_argument("--amp-policy", default=None,
                        help="New AmpPolicy value (Promote|KeepInput|ForceFP32)")
    parser.add_argument("--deterministic", action="store_true", default=None)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--version", type=int, default=None,
                        help="Schema version integer (default: keep existing)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "." not in args.op_name:
        parser.error("op_name must be family.snake_name (e.g. ufunc.cube_root)")
    family, snake = args.op_name.split(".", 1)
    camel = _snake_to_camel(snake)
    cpp_path = OPS_DIR / family / f"{camel}.cpp"

    if not cpp_path.exists():
        print(f"[regen_op] ERROR: {cpp_path} does not exist.")
        return 1

    # Read current values from the .cpp to use as defaults
    text = cpp_path.read_text()
    cur_amp = "Promote"
    cur_det = True
    cur_ver = 1

    amp_match = re.search(r'AmpPolicy::(\w+)', text)
    if amp_match:
        cur_amp = amp_match.group(1)
    det_match = re.search(r'/\*deterministic=\*/(true|false)', text)
    if det_match:
        cur_det = det_match.group(1) == "true"
    ver_match = re.search(rf'"{snake}",\s*(\d+)', text)
    if ver_match:
        cur_ver = int(ver_match.group(1))

    # Apply overrides
    amp = _parse_amp(args.amp_policy) if args.amp_policy else cur_amp
    det = args.deterministic if args.deterministic is not None else cur_det
    version = args.version if args.version is not None else cur_ver

    return regen(family, snake, amp, det, version, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
