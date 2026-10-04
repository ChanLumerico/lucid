"""lizard: long, branchy, and over-parameterised functions in Python and C++."""

import re
from typing import override

import lizard

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector
from tools.quality_gate.core import Counts, Scope


def _short_name(name: str) -> str:
    return re.split(r"::|\.", name)[-1]


class LizardCollector(Collector):
    name = "lizard"
    local = True
    fast = True

    @override
    def version(self) -> str:
        return str(getattr(lizard, "version", "?"))

    @override
    def collect(self, scope: Scope) -> Counts:
        counts: Counts = {}
        for path in scope.files:
            info = lizard.analyze_file(str(scope.root / path))
            is_py = path.endswith(".py")
            rules = {"function-length": 0, "function-ccn": 0, "internal-args": 0}
            for fn in info.function_list:
                if fn.length > config.FUNCTION_LENGTH_MAX:
                    rules["function-length"] += 1
                if fn.cyclomatic_complexity > config.FUNCTION_CCN_MAX:
                    rules["function-ccn"] += 1
                if (
                    is_py
                    and _short_name(fn.name).startswith("_")
                    and not _short_name(fn.name).startswith("__")
                    and fn.parameter_count > config.INTERNAL_ARGS_MAX
                ):
                    rules["internal-args"] += 1
            for rule, n in rules.items():
                if n:
                    counts[("lizard", rule, path)] = n
        return counts
