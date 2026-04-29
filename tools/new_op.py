#!/usr/bin/env python3
"""Phase 6.5 — Op scaffolding CLI.

Creates all boilerplate for a new Lucid op in one command:
  - lucid/_C/ops/<family>/<CamelName>.h  — schema + kernel declaration
  - lucid/_C/ops/<family>/<CamelName>.cpp — stub implementation
  - Patches CMakeLists.txt source list
  - Adds xfail parity spec to tests/parity/specs_<family>.py
  - Adds CHANGELOG.md entry

Usage examples:
  python tools/new_op.py ufunc.my_relu --kind unary --amp-policy KeepInput
  python tools/new_op.py bfunc.my_add  --kind binary --amp-policy Promote
  python tools/new_op.py ufunc.my_sum  --kind reduction --amp-policy ForceFP32
  python tools/new_op.py nn.my_layer   --kind nary --inputs 3 --amp-policy Promote
  python tools/new_op.py --validate
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent, indent

ROOT = Path(__file__).resolve().parent.parent
OPS_DIR  = ROOT / "lucid" / "_C" / "ops"
CMAKE    = ROOT / "lucid" / "_C" / "CMakeLists.txt"
TESTS    = ROOT / "tests" / "parity"
CHANGELOG = ROOT / "CHANGELOG.md"


# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

def _snake_to_camel(s: str) -> str:
    return "".join(w.capitalize() for w in s.split("_"))

def _family_subdir(family: str) -> Path:
    """ops/ufunc, ops/bfunc, ops/nn, etc."""
    return OPS_DIR / family

def _cmake_target(family: str) -> str:
    """The lucid_object_library target that owns this family."""
    mapping = {
        "ufunc": "lucid_ops",
        "bfunc": "lucid_ops",
        "utils": "lucid_ops",
        "gfunc": "lucid_ops",
        "linalg": "lucid_ops",
        "einops": "lucid_ops",
        "nn": "lucid_ops",
    }
    return mapping.get(family, "lucid_ops")

def _relative_cpp_path(family: str, camel: str) -> str:
    """Path relative to CMakeLists.txt (e.g. ops/ufunc/MyOp.cpp)."""
    return f"ops/{family}/{camel}.cpp"


# ---------------------------------------------------------------------------
# Template generators
# ---------------------------------------------------------------------------

UNARY_INCLUDE = {
    "ufunc": '../../',
    "bfunc": '../../',
    "nn":    '../',
    "linalg": '../',
}

def _depth_prefix(family: str) -> str:
    """../../ for ops/ufunc/, ../ for ops/nn/ etc."""
    depths = {"ufunc": "../../", "bfunc": "../../", "utils": "../../",
              "gfunc": "../../", "einops": "../../", "linalg": "../../",
              "nn": "../"}
    return depths.get(family, "../../")


def _gen_unary_header(snake: str, camel: str, family: str,
                      amp: str, det: bool, desc: str) -> str:
    pfx = _depth_prefix(family)
    det_str = "true" if det else "false"
    return dedent(f"""\
        #pragma once

        // {desc or f"Lucid — {snake} unary op."}

        #include "{pfx}api.h"
        #include "{pfx}backend/IBackend.h"
        #include "{pfx}core/AmpPolicy.h"
        #include "{pfx}core/OpSchema.h"
        #include "{pfx}core/Storage.h"
        #include "{pfx}core/fwd.h"
        #include "_UnaryOp.h"

        namespace lucid {{

        class LUCID_API {camel}Backward : public UnaryOp<{camel}Backward> {{
        public:
            static const OpSchema schema_v1;
            static Storage dispatch(backend::IBackend& be, const Storage& a,
                                    const Shape& s, Dtype dt) {{
                // TODO: call be.<method>(a, s, dt)
                (void)be; (void)a; (void)s; (void)dt;
                ErrorBuilder(schema_v1.name).not_implemented("dispatch not yet implemented");
            }}
            Storage grad_formula(const Storage& g);
        }};

        LUCID_API TensorImplPtr {snake}_op(const TensorImplPtr& a);

        }}  // namespace lucid
    """)


def _gen_unary_cpp(snake: str, camel: str, family: str,
                   amp: str, det: bool) -> str:
    pfx = _depth_prefix(family)
    det_str = "true" if det else "false"
    return dedent(f"""\
        #include "{camel}.h"

        #include "{pfx}core/Error.h"
        #include "{pfx}core/ErrorBuilder.h"
        #include "{pfx}core/OpRegistry.h"

        namespace lucid {{

        const OpSchema {camel}Backward::schema_v1{{
            "{snake}", 1, AmpPolicy::{amp}, /*deterministic=*/{det_str}}};

        Storage {camel}Backward::grad_formula(const Storage& g) {{
            // TODO: implement backward for {snake}
            (void)g;
            ErrorBuilder("{snake}").not_implemented("grad_formula not yet implemented");
        }}

        TensorImplPtr {snake}_op(const TensorImplPtr& a) {{
            return {camel}Backward::forward(a);
        }}
        LUCID_REGISTER_OP({camel}Backward)

        }}  // namespace lucid
    """)


def _gen_binary_header(snake: str, camel: str, family: str,
                       amp: str, det: bool, desc: str) -> str:
    pfx = _depth_prefix(family)
    return dedent(f"""\
        #pragma once

        // {desc or f"Lucid — {snake} binary op."}

        #include "{pfx}api.h"
        #include "{pfx}backend/IBackend.h"
        #include "{pfx}core/AmpPolicy.h"
        #include "{pfx}core/OpSchema.h"
        #include "{pfx}core/Storage.h"
        #include "{pfx}core/fwd.h"
        #include "_BinaryOp.h"

        namespace lucid {{

        class LUCID_API {camel}Backward : public BinaryOp<{camel}Backward> {{
        public:
            static const OpSchema schema_v1;
            static Storage dispatch(backend::IBackend& be,
                                    const Storage& a, const Storage& b,
                                    const Shape& s, Dtype dt) {{
                // TODO: call be.<method>(a, b, s, dt)
                (void)be; (void)a; (void)b; (void)s; (void)dt;
                ErrorBuilder(schema_v1.name).not_implemented("dispatch not yet implemented");
            }}
            std::pair<Storage, Storage> grad_formula(const Storage& g);
        }};

        LUCID_API TensorImplPtr {snake}_op(const TensorImplPtr& a, const TensorImplPtr& b);

        }}  // namespace lucid
    """)


def _gen_binary_cpp(snake: str, camel: str, family: str,
                    amp: str, det: bool) -> str:
    pfx = _depth_prefix(family)
    det_str = "true" if det else "false"
    return dedent(f"""\
        #include "{camel}.h"

        #include "{pfx}core/Error.h"
        #include "{pfx}core/ErrorBuilder.h"
        #include "{pfx}core/OpRegistry.h"

        namespace lucid {{

        const OpSchema {camel}Backward::schema_v1{{
            "{snake}", 1, AmpPolicy::{amp}, /*deterministic=*/{det_str}}};

        std::pair<Storage, Storage> {camel}Backward::grad_formula(const Storage& g) {{
            // TODO: implement backward for {snake}
            (void)g;
            ErrorBuilder("{snake}").not_implemented("grad_formula not yet implemented");
        }}

        TensorImplPtr {snake}_op(const TensorImplPtr& a, const TensorImplPtr& b) {{
            return {camel}Backward::forward(a, b);
        }}
        LUCID_REGISTER_OP({camel}Backward)

        }}  // namespace lucid
    """)


def _gen_reduction_header(snake: str, camel: str, family: str,
                          amp: str, det: bool, desc: str) -> str:
    pfx = _depth_prefix(family)
    return dedent(f"""\
        #pragma once

        // {desc or f"Lucid — {snake} reduction op."}

        #include <vector>

        #include "{pfx}api.h"
        #include "{pfx}backend/IBackend.h"
        #include "{pfx}core/AmpPolicy.h"
        #include "{pfx}core/OpSchema.h"
        #include "{pfx}core/Storage.h"
        #include "{pfx}core/fwd.h"
        #include "_ReduceOp.h"

        namespace lucid {{

        class LUCID_API {camel}Backward : public ReduceOp<{camel}Backward> {{
        public:
            static const OpSchema schema_v1;
            static Storage dispatch(backend::IBackend& be, const Storage& a,
                                    const Shape& in_shape,
                                    const std::vector<int>& axes, bool keepdims, Dtype dt) {{
                // TODO: call be.<reduce_method>(a, in_shape, {{axes, keepdims}}, dt)
                (void)be; (void)a; (void)in_shape; (void)axes; (void)keepdims; (void)dt;
                ErrorBuilder(schema_v1.name).not_implemented("dispatch not yet implemented");
            }}
            Storage grad_formula(const Storage& g);
        }};

        LUCID_API TensorImplPtr {snake}_op(const TensorImplPtr& a,
                                           const std::vector<int>& axes, bool keepdims);

        }}  // namespace lucid
    """)


def _gen_reduction_cpp(snake: str, camel: str, family: str,
                       amp: str, det: bool) -> str:
    pfx = _depth_prefix(family)
    det_str = "true" if det else "false"
    return dedent(f"""\
        #include "{camel}.h"

        #include "{pfx}core/Error.h"
        #include "{pfx}core/ErrorBuilder.h"
        #include "{pfx}core/OpRegistry.h"

        namespace lucid {{

        const OpSchema {camel}Backward::schema_v1{{
            "{snake}", 1, AmpPolicy::{amp}, /*deterministic=*/{det_str}}};

        Storage {camel}Backward::grad_formula(const Storage& g) {{
            // TODO: implement backward for {snake}
            (void)g;
            ErrorBuilder("{snake}").not_implemented("grad_formula not yet implemented");
        }}

        TensorImplPtr {snake}_op(const TensorImplPtr& a,
                                  const std::vector<int>& axes, bool keepdims) {{
            return {camel}Backward::forward(a, axes, keepdims);
        }}
        LUCID_REGISTER_OP({camel}Backward)

        }}  // namespace lucid
    """)


def _gen_nary_header(snake: str, camel: str, family: str,
                     n_inputs: int, amp: str, det: bool, desc: str) -> str:
    pfx = _depth_prefix(family)
    args = ", ".join(f"const TensorImplPtr& x{i}" for i in range(n_inputs))
    return dedent(f"""\
        #pragma once

        // {desc or f"Lucid — {snake} N-ary op ({n_inputs} inputs)."}

        #include <array>
        #include <memory>

        #include "{pfx}api.h"
        #include "{pfx}autograd/AutogradNode.h"
        #include "{pfx}core/AmpPolicy.h"
        #include "{pfx}core/OpSchema.h"
        #include "{pfx}core/Storage.h"
        #include "{pfx}core/fwd.h"
        #include "{pfx}kernel/NaryKernel.h"

        namespace lucid {{

        class LUCID_API {camel}Backward : public AutogradNode<{camel}Backward, {n_inputs}> {{
        public:
            static const OpSchema schema_v1;
            static TensorImplPtr forward({args});
            std::vector<Storage> apply(Storage grad_out) override;
        }};

        LUCID_API TensorImplPtr {snake}_op({args});

        }}  // namespace lucid
    """)


def _gen_nary_cpp(snake: str, camel: str, family: str,
                  n_inputs: int, amp: str, det: bool) -> str:
    pfx = _depth_prefix(family)
    det_str = "true" if det else "false"
    params = ", ".join(f"const TensorImplPtr& x{i}" for i in range(n_inputs))
    arr_init = ", ".join(f"x{i}" for i in range(n_inputs))
    return dedent(f"""\
        #include "{camel}.h"

        #include "{pfx}core/Error.h"
        #include "{pfx}core/ErrorBuilder.h"
        #include "{pfx}core/OpRegistry.h"
        #include "{pfx}core/SchemaGuard.h"
        #include "{pfx}core/Scope.h"

        namespace lucid {{

        const OpSchema {camel}Backward::schema_v1{{
            "{snake}", 1, AmpPolicy::{amp}, /*deterministic=*/{det_str}}};

        TensorImplPtr {camel}Backward::forward({params}) {{
            // TODO: implement forward for {snake}
            ErrorBuilder("{snake}").not_implemented("forward not yet implemented");
        }}

        std::vector<Storage> {camel}Backward::apply(Storage grad_out) {{
            // TODO: implement backward for {snake}
            (void)grad_out;
            ErrorBuilder("{snake}").not_implemented("apply not yet implemented");
        }}

        TensorImplPtr {snake}_op({params}) {{
            return {camel}Backward::forward({arr_init});
        }}
        LUCID_REGISTER_OP({camel}Backward)

        }}  // namespace lucid
    """)


# ---------------------------------------------------------------------------
# Parity spec stub
# ---------------------------------------------------------------------------

def _gen_parity_stub(snake: str, family: str) -> str:
    """Return (stub_text, target_file).

    Stubs go into tests/parity/test_scaffolded.py (created if needed)
    so they don't pollute specs_*.py data files.
    """
    stub_file = TESTS / "test_scaffolded.py"
    stub = dedent(f"""\


        # ---- {snake} (scaffolded by tools/new_op.py) --------------------
        @pytest.mark.xfail(reason="{snake} not yet implemented", strict=True)
        @pytest.mark.parametrize("device", [E.Device.CPU, E.Device.GPU])
        def test_{snake}_forward(device):
            import numpy as np
            x = np.random.randn(4, 5).astype("float32")
            t = E.TensorImpl(x, device, False)
            out = E.{snake}(t)
            assert out is not None
    """)
    return stub, stub_file


# ---------------------------------------------------------------------------
# CMakeLists.txt patching
# ---------------------------------------------------------------------------

def _patch_cmake(family: str, camel: str) -> bool:
    """Insert the new .cpp source into the lucid_ops target in CMakeLists.txt."""
    text = CMAKE.read_text()
    rel = f"ops/{family}/{camel}.cpp"
    if rel in text:
        return False  # already present

    # Find last entry of ops/<family>/ in the lucid_ops block and insert after it.
    pattern = re.compile(rf"(    ops/{family}/\S+\.cpp)")
    matches = list(pattern.finditer(text))
    if matches:
        last = matches[-1]
        insert_pos = last.end()
        new_text = text[:insert_pos] + f"\n    {rel}" + text[insert_pos:]
    else:
        # Family not present — append before closing paren of lucid_ops block.
        # Find lucid_ops block end (the closing paren line)
        # Simple heuristic: insert before "    ops/gfunc/" block or before closing )
        insert_marker = "    ops/gfunc/"
        idx = text.find(insert_marker)
        if idx == -1:
            # fallback: just before last ) in lucid_ops
            idx = text.rfind("    ops/")
            idx = text.find("\n", idx) + 1
        new_text = text[:idx] + f"    {rel}\n" + text[idx:]

    CMAKE.write_text(new_text)
    return True


# ---------------------------------------------------------------------------
# CHANGELOG patching
# ---------------------------------------------------------------------------

def _patch_changelog(snake: str, kind: str, desc: str) -> None:
    text = CHANGELOG.read_text()
    entry = f"- Scaffolded `{snake}` ({kind}){': ' + desc if desc else ''}. (tools/new_op.py)\n"
    marker = "## [Unreleased]"
    idx = text.find(marker)
    if idx == -1:
        return
    # Find the first "### Added" section after [Unreleased]
    added_idx = text.find("### Added", idx)
    if added_idx == -1:
        # Insert new ### Added section
        insert_after = text.find("\n", idx) + 1
        new_text = text[:insert_after] + "\n### Added\n" + entry + text[insert_after:]
    else:
        insert_after = text.find("\n", added_idx) + 1
        new_text = text[:insert_after] + entry + text[insert_after:]
    CHANGELOG.write_text(new_text)


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------

def _validate() -> int:
    """Check that every registered op has a .h+.cpp file and CMakeLists entry."""
    sys.path.insert(0, str(ROOT))
    from lucid._C import engine as E

    schemas = {s.name: s for s in E.op_registry_all()}
    cmake_text = CMAKE.read_text()
    errors = 0

    # Check that all .cpp files referenced in CMakeLists exist.
    cpp_refs = re.findall(r"ops/\S+\.cpp", cmake_text)
    for ref in cpp_refs:
        full = ROOT / "lucid" / "_C" / ref
        if not full.exists():
            print(f"[MISSING FILE] {full}")
            errors += 1

    print(f"[validate] {len(schemas)} schemas, {len(cpp_refs)} CMakeLists entries, "
          f"{errors} errors")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_kind_amp(kind: str, amp_arg: str) -> str:
    """Normalise AmpPolicy string."""
    mapping = {
        "promote": "Promote",
        "keepinput": "KeepInput",
        "keep_input": "KeepInput",
        "forcefp32": "ForceFP32",
        "force_fp32": "ForceFP32",
    }
    return mapping.get(amp_arg.lower(), amp_arg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("op_name", nargs="?",
                        help="Op name in family.snake_name format (e.g. ufunc.my_relu)")
    parser.add_argument("--kind", choices=["unary", "binary", "reduction", "nary"],
                        default="unary")
    parser.add_argument("--inputs", type=int, default=1,
                        help="Number of tensor inputs (for --kind nary).")
    parser.add_argument("--amp-policy", default="Promote",
                        help="AmpPolicy: Promote | KeepInput | ForceFP32")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--description", default="", help="One-line op description.")
    parser.add_argument("--validate", action="store_true",
                        help="Validate schema vs files, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be generated without writing files.")

    args = parser.parse_args()

    if args.validate:
        return _validate()

    if not args.op_name:
        parser.error("op_name is required unless --validate is passed.")

    # Parse "family.snake_name"
    if "." not in args.op_name:
        parser.error("op_name must be in family.snake_name format (e.g. ufunc.my_relu)")
    family, snake = args.op_name.split(".", 1)
    camel = _snake_to_camel(snake)
    amp = _parse_kind_amp(args.kind, args.amp_policy)
    det = args.deterministic
    kind = args.kind
    n_inputs = args.inputs if kind == "nary" else (1 if kind in ("unary", "reduction") else 2)

    subdir = _family_subdir(family)
    h_path = subdir / f"{camel}.h"
    cpp_path = subdir / f"{camel}.cpp"

    # Generate file contents
    if kind == "unary":
        h_content = _gen_unary_header(snake, camel, family, amp, det, args.description)
        cpp_content = _gen_unary_cpp(snake, camel, family, amp, det)
    elif kind == "binary":
        h_content = _gen_binary_header(snake, camel, family, amp, det, args.description)
        cpp_content = _gen_binary_cpp(snake, camel, family, amp, det)
    elif kind == "reduction":
        h_content = _gen_reduction_header(snake, camel, family, amp, det, args.description)
        cpp_content = _gen_reduction_cpp(snake, camel, family, amp, det)
    else:  # nary
        h_content = _gen_nary_header(snake, camel, family, n_inputs, amp, det, args.description)
        cpp_content = _gen_nary_cpp(snake, camel, family, n_inputs, amp, det)

    parity_stub, spec_file = _gen_parity_stub(snake, family)

    if args.dry_run:
        print(f"\n=== {h_path} ===")
        print(h_content)
        print(f"\n=== {cpp_path} ===")
        print(cpp_content)
        print(f"\n=== append to {spec_file} ===")
        print(parity_stub)
        print(f"\n=== CMakeLists.txt +  ops/{family}/{camel}.cpp ===")
        print(f"\n=== CHANGELOG.md + entry for {snake} ===")
        return 0

    # Safety check — don't overwrite
    if h_path.exists() or cpp_path.exists():
        print(f"[error] {h_path} or {cpp_path} already exists. Aborting.")
        return 1

    subdir.mkdir(parents=True, exist_ok=True)
    h_path.write_text(h_content)
    cpp_path.write_text(cpp_content)
    print(f"[new_op] created {h_path.relative_to(ROOT)}")
    print(f"[new_op] created {cpp_path.relative_to(ROOT)}")

    # CMakeLists.txt
    patched = _patch_cmake(family, camel)
    if patched:
        print(f"[new_op] patched CMakeLists.txt  (+ops/{family}/{camel}.cpp)")
    else:
        print(f"[new_op] CMakeLists.txt already contains {camel}.cpp — skipped")

    # Parity spec — always goes to test_scaffolded.py
    if not spec_file.exists():
        header = dedent("""\
            \"\"\"Scaffolded op stubs — auto-generated by tools/new_op.py.
            Each test is xfail until the op is fully implemented.
            \"\"\"
            from __future__ import annotations
            import pytest
            from lucid._C import engine as E
        """)
        spec_file.write_text(header)
    with open(spec_file, "a") as f:
        f.write(parity_stub)
    print(f"[new_op] appended stub to {spec_file.relative_to(ROOT)}")

    # CHANGELOG
    _patch_changelog(snake, kind, args.description)
    print(f"[new_op] updated CHANGELOG.md")

    print(f"\n✓ {snake} scaffolded. Next steps:")
    print(f"  1. Implement dispatch() in {h_path.relative_to(ROOT)}")
    print(f"  2. Implement grad_formula() in {cpp_path.relative_to(ROOT)}")
    print(f"  3. Wire binding in lucid/_C/bindings/bind_{family}.cpp")
    print(f"  4. Run: python setup.py build_ext --inplace")
    print(f"  5. Flip xfail → passing in {spec_file.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
