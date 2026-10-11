"""THIRD_PARTY_NOTICES.md describes the MLX build the wheel actually ships.

The wheel carries ``liblucid_mlx`` (``pyproject.toml [tool.lucid.mlx]``) and,
as its notice, the static ``THIRD_PARTY_NOTICES.md`` — the staged LICENSE is
not shipped.  So the notice has to name the pinned version, wheel and sha256,
and reproduce that wheel's LICENSE byte for byte.  A pin bump that forgets
either would ship a notice for a binary that is not in the wheel.
"""

import re
import tomllib
from pathlib import Path

import pytest

import lucid

ROOT = Path(lucid.__file__).resolve().parent.parent
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
PYPROJECT = ROOT / "pyproject.toml"


def _pin() -> dict[str, str]:
    if not PYPROJECT.is_file() or not NOTICES.is_file():
        pytest.skip("not a source checkout (installed wheel)")
    with PYPROJECT.open("rb") as fh:
        return dict(tomllib.load(fh)["tool"]["lucid"]["mlx"])


def _mlx_section() -> str:
    text = NOTICES.read_text()
    start = text.index("\n## MLX\n")
    return text[start : text.index("\n### ", start)]


def test_the_mlx_section_names_the_pinned_build() -> None:
    pin = _pin()
    section = _mlx_section()
    assert pin["wheel"] in section
    assert pin["sha256"] in section
    assert f"mlx_metal-{pin['version']}.dist-info" in section
    versions = set(re.findall(r"\bv(\d+\.\d+\.\d+)\b", section))
    assert versions == {pin["version"]}, f"MLX versions named: {sorted(versions)}"


def test_the_mlx_license_is_the_pinned_wheels_verbatim() -> None:
    pin = _pin()
    # The entry the engine was built against; staged on this machine by any
    # build.  Imported lazily: tools/ is a source-checkout directory.
    from tools import fetch_mlx

    license_file = fetch_mlx.pinned_entry(fetch_mlx.read_pin()) / "LICENSE"
    if not license_file.is_file():
        pytest.skip(f"MLX {pin['version']} is not staged here with its LICENSE")
    block = re.search(r"```text\n(.*?)```\n", _mlx_section(), re.S)
    assert block is not None, "the MLX section has no ```text license block"
    assert block.group(1) == license_file.read_text(), (
        "THIRD_PARTY_NOTICES.md's MLX license differs from the pinned wheel's; "
        f"copy {license_file} into it byte for byte"
    )
