"""Nothing in ``lucid/`` reaches for the ``mlx`` Python package by accident.

The engine runs on its own pinned MLX (``liblucid_mlx``, LCD-292).  A Python
``mlx`` in the same process is a second MLX instance with its own allocator,
buffer cache and command queue, so asking it about memory, the cache or
synchronisation answers for the wrong instance.  Two tests did exactly that
after the switch: a peak-memory comparison read ``0 < 0`` and a leak probe
read zero drift whether or not it leaked.  The engine binds every such query
(``metal_active_memory``, ``metal_peak_memory``, ``metal_cache_memory``,
``synchronize_gpu``, ...), so the only reason left to import ``mlx`` is to
exercise the user's package on purpose.  This test keeps it that way: an
import of ``mlx`` anywhere else — including in a subprocess script held in a
string — fails here.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # lucid/

# Places that import the user's `mlx` on purpose.
_ALLOWED = (
    # Interop with the `mlx` package is the subject under test.
    ROOT / "test/unit/interop",
    # Opt-in speed comparisons against `mlx` itself, never run by the suite.
    ROOT / "benchmarks",
)

_IMPORT = re.compile(
    r"(?:^|[\s;])(?:import\s+mlx\b|from\s+mlx\b)"
    r"|(?:importorskip|import_module|find_spec)\(\s*[\"']mlx\b"
)


def _offenders() -> list[str]:
    hits = []
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or any(path.is_relative_to(a) for a in _ALLOWED):
            continue
        for number, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if _IMPORT.search(line):
                hits.append(f"{path.relative_to(ROOT.parent)}:{number}: {line.strip()[:100]}")
    return hits


def test_only_interop_tests_import_the_mlx_package() -> None:
    hits = _offenders()
    assert not hits, (
        "the `mlx` package is a different MLX instance from the engine's; use the "
        "engine bindings (lucid._C.engine.metal_* / synchronize_gpu):\n" + "\n".join(hits)
    )


def test_the_rule_sees_imports_inside_subprocess_scripts() -> None:
    # Assembled so that this file does not trip its own rule.
    package = "ml" + "x"
    assert _IMPORT.search(f"import gc, sys; import {package}.core as mx")
    assert _IMPORT.search(f"    from {package}.core import array")
    assert _IMPORT.search(f'    pytest.importorskip("{package}.core")')
    assert not _IMPORT.search(f"    an ``{package}.core.array`` is accepted")
