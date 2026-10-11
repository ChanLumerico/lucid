"""Per-directory pytest config for ``lucid/test/unit/models``.

Pending families declare an explicit module-level skip in their test file.
Directory collect_ignore is deliberately not used: pytest bypasses it for
explicit file arguments, which the memory-bounded audit runner supplies.
Explicit skips remain visible and activate when the family is implemented.
"""

import sys
from collections.abc import Callable

import pytest

import lucid

type HostReads = Callable[[Callable[[], object]], list[str]]


@pytest.fixture
def host_reads(monkeypatch: pytest.MonkeyPatch) -> HostReads:
    """Run a callable and return ``file:line`` of every ``.item()`` /
    ``.tolist()`` / ``.numpy()`` that model code under ``lucid/models`` made.

    Each is a device sync on Metal.  Reads made by the framework's own
    index guards are not the model's and are not counted.
    """
    seen: list[str] = []
    for name in ("item", "tolist", "numpy"):
        original = getattr(lucid.Tensor, name)

        def spy(t: lucid.Tensor, *a: object, _f=original, **k: object) -> object:
            caller = sys._getframe(1)
            if "lucid/models/" in caller.f_code.co_filename:
                seen.append(f"{caller.f_code.co_filename}:{caller.f_lineno}")
            return _f(t, *a, **k)

        monkeypatch.setattr(lucid.Tensor, name, spy)

    def count(run: Callable[[], object]) -> list[str]:
        seen.clear()
        run()
        return list(seen)

    return count
