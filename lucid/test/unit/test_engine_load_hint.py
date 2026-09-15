"""The engine names a known cause when it cannot load.

MLX 0.32's macOS 26 build is compiled for macOS 26.2, so on 26.0 and 26.1
the engine's import can fail inside ``libmlx.dylib``.  The hint says so,
and stays out of the way everywhere else.
"""

import pytest

from lucid._C import _mlx_load_hint


@pytest.mark.parametrize("macos", ["26.0", "26", "26.1", "26.1.1"])
def test_mlx_0_32_on_macos_26_before_26_2_gets_a_hint(macos: str) -> None:
    hint = _mlx_load_hint(macos, "0.32.2")
    assert hint is not None
    assert "26.2" in hint
    assert "mlx<0.32" in hint


@pytest.mark.parametrize(
    ("macos", "mlx"),
    [
        ("26.2", "0.32.2"),
        ("27.0", "0.32.2"),
        ("15.5", "0.32.2"),
        ("26.1", "0.31.2"),
        ("26.1", None),
        ("", "0.32.2"),
    ],
)
def test_no_hint_where_that_cause_cannot_apply(macos: str, mlx: str | None) -> None:
    assert _mlx_load_hint(macos, mlx) is None
