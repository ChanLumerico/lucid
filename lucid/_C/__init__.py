"""The compiled engine, imported here as ``_C_engine``.

The running macOS is checked against the engine's floor before the engine
loads, and a load failure with a known cause says how to recover.
"""

import platform

# The minos of the MLX build the engine ships (pyproject.toml
# [tool.lucid.mlx] macos).  No wheel tag can carry it: pip treats every macOS
# 11+ release as X.0, so a 26_0 wheel installs on 26.0 and 26.1 too.
_MACOS_FLOOR = (26, 2)


def _version(text: str) -> tuple[int, ...]:
    """The leading ``major.minor`` of a version string, as integers."""
    return tuple(int(p) for p in text.split(".")[:2] if p.isdigit())


def _macos_floor_error(macos: str) -> str | None:
    """Why the engine cannot run on *macos*, or ``None`` when it can.

    Parameters
    ----------
    macos : str
        The running macOS version, as ``platform.mac_ver()[0]`` gives it
        (empty off macOS, where the engine's own guards apply).

    Returns
    -------
    str or None
        The import error message, or ``None`` at or above the floor.
    """
    running = _version(macos)
    if not running or running >= _MACOS_FLOOR:
        return None
    floor = ".".join(map(str, _MACOS_FLOOR))
    return (
        f"Lucid needs macOS {floor} or later; this is macOS {macos}. The MLX build "
        f"bundled with its engine is compiled for {floor}. Update macOS."
    )


def _engine_load_hint(message: str) -> str | None:
    """A recovery step for a known engine load failure, or ``None``.

    Parameters
    ----------
    message : str
        The text of the ``ImportError`` the engine's import raised.

    Returns
    -------
    str or None
        The hint to append, or ``None`` when the cause is not a known one.
    """
    if "liblucid_mlx.dylib" not in message:
        return None
    return (
        "The engine's bundled MLX (liblucid_mlx.dylib) is missing. In a source "
        "checkout, run `python3 tools/fetch_mlx.py` on this machine: it re-stages "
        "the per-machine cache entry the engine was linked against, with no "
        "rebuild. In an installed wheel, reinstall lucid-dl."
    )


_floor_error = _macos_floor_error(platform.mac_ver()[0])
if _floor_error is not None:
    raise ImportError(_floor_error)

try:
    from lucid._C import engine as _C_engine
except ImportError as exc:
    _hint = _engine_load_hint(str(exc))
    if _hint is None:
        raise
    raise ImportError(f"{exc}\n\n{_hint}") from exc
