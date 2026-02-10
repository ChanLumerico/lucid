from pathlib import Path
import json

from lucid import models
from lucid.weights import _classname


reg_path = Path("lucid/weights/registry.json")
out_path = Path("lucid/weights/__init__.pyi")

reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}

basic_stub = [
    "from enum import Enum",
    "from dataclasses import dataclass",
    "from typing import Any, Dict, Optional",
    "",
    "@dataclass(frozen=True)",
    "class WeightEntry:",
    "    url: str",
    "    sha256: str",
    "    tag: str",
    "    dataset: Optional[str] = None",
    "    meta: Optional[Dict[str, Any]] = None",
    "    @property",
    "    def config(self) -> Optional[Dict[str, Any]]: ...",
    "",
]

lines = []
lines.extend(basic_stub)

all_names = []
for key, entries in reg.items():
    default_meta = entries.get("DEFAULT", {}).get("meta", {})
    name = _classname(
        key,
        family=default_meta.get("family", None),
        enum_name=default_meta.get("enum_name", None),
    )
    if name is None:
        raise RuntimeError(f"Cannot resolve model key '{key}'.")

    all_names.append(name)
    lines.append(f"class {name}(Enum):")

    for tag in entries.keys():
        lines.append(f"    {tag}: WeightEntry")
    lines.append("")

lines.append("__all__ = [")
for name in all_names:
    lines.append(f'    "{name}",')
lines.append("]")
lines.append("")

out_path.write_text("\n".join(lines) + "")
