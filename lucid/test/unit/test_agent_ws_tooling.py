"""Expose ``tools/test_agent_ws.py`` under ``lucid/test/`` so default pytest CI collects it (LCD-30)."""

import importlib.util
from pathlib import Path

_TEST_FILE = Path(__file__).resolve().parents[3] / "tools" / "test_agent_ws.py"
_SPEC = importlib.util.spec_from_file_location("_tools_test_agent_ws", _TEST_FILE)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

for _name in dir(_MOD):
    if _name == "scratch" or _name.startswith("test_"):
        globals()[_name] = getattr(_MOD, _name)
