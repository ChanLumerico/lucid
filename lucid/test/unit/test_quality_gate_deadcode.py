"""Unit tests for the vulture dead-code collector and its whitelist (DS-15 / LCD-356)."""

from pathlib import Path

import pytest

from tools.quality_gate.collectors.deadcode import (
    _IGNORE_DECORATORS,
    WHITELIST,
    VultureCollector,
)
from tools.quality_gate.core import Scope


def _collect_single_file(tmp_path: Path, source: str) -> dict[tuple[str, str, str], int]:
    pkg = tmp_path / "lucid"
    pkg.mkdir(parents=True, exist_ok=True)
    target = pkg / "sample.py"
    target.write_text(source)
    scope = Scope(
        root=tmp_path,
        py=["lucid/sample.py"],
        cpp=[],
        workdir=tmp_path,
    )
    return VultureCollector().collect(scope)


@pytest.mark.parametrize(
    "decorator",
    [
        "@_emitter('add')",
        "@register_kl(int, int)",
        "@register_model('my_model')",
        "@register_weights(object)",
        "@_register_preset",
        "@_register(frozenset({'zeros'}))",
    ],
)
def test_registry_decorators_are_not_flagged_while_unused_function_is(
    tmp_path: Path,
    decorator: str,
) -> None:
    """Functions registered via decorator callbacks must not count as dead code."""
    src = (
        f"{decorator}\n"
        "def _registered_callback() -> int:\n"
        "    return 1\n\n"
        "def _genuinely_unused_function() -> int:\n"
        "    return 2\n"
    )
    counts = _collect_single_file(tmp_path, src)
    assert counts == {("vulture", "unused-function", "lucid/sample.py"): 1}


@pytest.mark.parametrize(
    ("kind", "snippet", "dead_snippet", "expected_rule"),
    [
        (
            "lazy_loader",
            "def _load_factories() -> dict[str, object]:\n    return {}\n",
            "def _truly_dead_loader() -> dict[str, object]:\n    return {}\n",
            "unused-function",
        ),
        (
            "tensor_public_method",
            "class Tensor:\n"
            "    def type_as(self) -> int:\n"
            "        return 0\n"
            "    def new_empty(self) -> int:\n"
            "        return 0\n"
            "    def new_tensor(self) -> int:\n"
            "        return 0\n"
            "    def detach_(self) -> int:\n"
            "        return 0\n"
            "Tensor()\n",
            "class _Holder:\n"
            "    def _truly_dead_method(self) -> int:\n"
            "        return 0\n"
            "_Holder()\n",
            "unused-method",
        ),
        (
            "weights_enum_members",
            "class Weights:\n"
            "    ALIEN = 1\n"
            "    UPNDOWN = 2\n"
            "    IMAGENET1K_512 = 3\n"
            "Weights()\n",
            "class _DeadWeights:\n"
            "    TRULY_DEAD_MEMBER = 99\n"
            "_DeadWeights()\n",
            "unused-variable",
        ),
        (
            "cpp_and_stdlib_hooks",
            "class Hooks:\n"
            "    def __init__(self) -> None:\n"
            "        self.once_differentiable = True\n"
            "        self.redraw_rng = False\n"
            "    def persistent_id(self) -> None:\n"
            "        return None\n"
            "    def _on_queue_feeder_error(self) -> None:\n"
            "        return None\n"
            "Hooks()\n",
            "class _AttrHolder:\n"
            "    def __init__(self) -> None:\n"
            "        self._truly_dead_attr = 1\n"
            "_AttrHolder()\n",
            "unused-attribute",
        ),
    ],
)
def test_whitelisted_indirect_symbols_are_not_flagged(
    tmp_path: Path,
    kind: str,
    snippet: str,
    dead_snippet: str,
    expected_rule: str,
) -> None:
    """Whitelisted public API, WeightsEnum members, lazy loaders, and hooks are ignored."""
    del kind
    assert WHITELIST.is_file()
    assert "@_emitter" in _IGNORE_DECORATORS
    assert "@register_kl" in _IGNORE_DECORATORS

    clean_counts = _collect_single_file(tmp_path, snippet)
    assert clean_counts == {}

    seeded_counts = _collect_single_file(tmp_path, snippet + "\n" + dead_snippet)
    assert seeded_counts == {("vulture", expected_rule, "lucid/sample.py"): 1}
