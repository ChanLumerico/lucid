"""clang-tidy on the engine — a stub, off by default; DS-3 (LCD-262) fills it.

It needs ``compile_commands.json`` (scripts/build_compile_commands.sh) and
minutes per run, so it will be opt-in (``--collectors clang-tidy``) until
DS-3 decides the check set and whether it joins ``--full``.
"""

from typing import override

from tools.quality_gate.collectors import Collector
from tools.quality_gate.core import Counts, GateError, Scope


class ClangTidyCollector(Collector):
    name = "clang-tidy"
    default = False
    in_diff = False

    @override
    def version(self) -> str:
        return "stub"

    @override
    def collect(self, scope: Scope) -> Counts:
        del scope  # the Collector signature; a stub measures nothing
        raise GateError("clang-tidy collector is a stub until DS-3 (LCD-262)")
