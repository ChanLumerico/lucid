"""lizard: long, branchy, and over-parameterised functions in Python and C++.

``function-length`` / ``function-ccn`` count the functions over a limit;
``function-length-excess`` / ``function-ccn-excess`` add up how far over
they are, so an already-long function cannot grow unseen (DS-29).
"""

import re
from typing import override

import lizard

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector
from tools.quality_gate.core import Counts, Scope

_RULES = (
    "function-length",
    "function-ccn",
    "internal-args",
    "function-length-excess",
    "function-ccn-excess",
)


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
            rules = dict.fromkeys(_RULES, 0)
            for fn in info.function_list:
                long_by = fn.length - config.FUNCTION_LENGTH_MAX
                branchy_by = fn.cyclomatic_complexity - config.FUNCTION_CCN_MAX
                if long_by > 0:
                    rules["function-length"] += 1
                    rules["function-length-excess"] += long_by
                if branchy_by > 0:
                    rules["function-ccn"] += 1
                    rules["function-ccn-excess"] += branchy_by
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
