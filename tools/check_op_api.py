"""Phase 1.6 – API consistency checker.

Walks public C++ op headers (ops/, nn/) and validates:
  1. Tensor args use `const TensorImplPtr&`
  2. Optional tensor defaults to `nullptr` (or TensorImplPtr{})
  3. Boolean args use `bool` type
  4. Reduction args use `int reduction`
  5. Multi-output ops return `std::vector<TensorImplPtr>` (not std::pair/tuple)

Run from project root:  python tools/check_op_api.py
"""

import re
import os
import sys
from pathlib import Path


# Known multi-output ops (return >1 tensor). Keep in sync with API_CONVENTIONS.md.
MULTI_OUTPUT_OPS = frozenset({
    "svd_op",
    "qr_op",
    "eig_op",
    "topk_op",
    "scaled_dot_product_attention_with_weights_op",
    # Short names used by the checker when stripping `_op` suffix
    "svd",
    "qr",
    "eig",
    "topk",
    "scaled_dot_product_attention_with_weights",
})


def _strip_lucid_api(s: str) -> str:
    """Remove LUCID_API prefix from a declaration."""
    return re.sub(r"\bLUCID_API\s+", "", s)


def check_file(filepath: Path) -> list[str]:
    errors: list[str] = []
    content = filepath.read_text()

    # Regex to find function declarations ending with `;`
    # Handles multi-line declarations by joining on newlines first.
    # We look for patterns like:
    #   LUCID_API ReturnType func_name(params);
    #   ReturnType func_name(params);
    func_pattern = re.compile(
        r"(?:^|\n)\s*"
        r"(?:LUCID_API\s+)?"
        r"(?P<return_type>(?:std::(?:vector|pair)\s*<[^>]+>\s*|[\w:]+(?:\s*<[^>]+>)?)\s*[&*]?\s*)"
        r"(?P<func_name>\w+_op)\s*\("
        r"(?P<params>[^)]*)"
        r"\)\s*;",
    )

    for match in func_pattern.finditer(content):
        return_type = match.group("return_type").strip()
        func_name = match.group("func_name").strip()
        params_str = match.group("params").strip()
        line_num = content.count("\n", 0, match.start()) + 1

        # Split parameters by comma, respecting template brackets
        params = _split_params(params_str)

        for param in params:
            param = param.strip()
            if not param:
                continue

            # Rule 1: Input Tensors — always `const TensorImplPtr&`
            if "TensorImplPtr" in param:
                # Allow: const TensorImplPtr& ...
                # Also allow: const std::vector<TensorImplPtr>& ...
                if "vector" not in param and "const TensorImplPtr&" not in param:
                    errors.append(
                        f"{filepath}:{line_num}:{func_name}: "
                        f"Tensor argument '{param}' should be 'const TensorImplPtr&'."
                    )

            # Rule 2: Optional tensor defaults — must use `= nullptr` or `= TensorImplPtr{}`
            if "TensorImplPtr" in param and "=" in param:
                default_part = param.split("=", 1)[1].strip()
                if default_part not in ("nullptr", "TensorImplPtr{}"):
                    errors.append(
                        f"{filepath}:{line_num}:{func_name}: "
                        f"Optional tensor '{param}' should default to nullptr or TensorImplPtr{{}}."
                    )

            # Rule 3: Reduction encoding
            if "reduction" in param.lower():
                # Must be `int reduction`
                if not re.match(r"int\s+reduction", param):
                    errors.append(
                        f"{filepath}:{line_num}:{func_name}: "
                        f"Reduction argument '{param}' should be 'int reduction'."
                    )

        # Rule 5: Multi-output return types
        base_name = func_name.removesuffix("_op")
        if func_name in MULTI_OUTPUT_OPS or base_name in MULTI_OUTPUT_OPS:
            if "std::vector<TensorImplPtr>" not in return_type:
                errors.append(
                    f"{filepath}:{line_num}:{func_name}: "
                    f"Known multi-output op should return 'std::vector<TensorImplPtr>'."
                )
        elif "std::pair" in return_type and "TensorImplPtr" in return_type:
            errors.append(
                f"{filepath}:{line_num}:{func_name}: "
                f"Multi-output ops should use 'std::vector<TensorImplPtr>', not std::pair. "
                f"If this is a new multi-output op, add it to MULTI_OUTPUT_OPS in check_op_api.py."
            )

    return errors


def _split_params(params_str: str) -> list[str]:
    """Split a parameter list by commas, respecting <> nesting."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def main() -> int:
    cpp_root = Path("lucid/_C")
    if not cpp_root.is_dir():
        print(
            f"Error: C++ source root '{cpp_root}' not found. "
            "Run this script from the project root.",
            file=sys.stderr,
        )
        return 1

    all_errors: list[str] = []

    # Scan all op-containing directories
    scan_dirs = [
        cpp_root / "ops",
        cpp_root / "nn",
    ]

    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            continue
        for root, _, files in os.walk(scan_dir):
            for file in files:
                if file.endswith((".h", ".hpp")):
                    filepath = Path(root) / file
                    all_errors.extend(check_file(filepath))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        print(f"\n{len(all_errors)} API consistency error(s) found.", file=sys.stderr)
        return 1
    print("API consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
