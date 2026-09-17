"""The compiled engine, imported here as ``_C_engine``.

When the engine cannot load, the error carries a hint if the cause is a
known one.
"""


def _version(text: str) -> tuple[int, ...]:
    """The leading ``major.minor`` of a version string, as integers."""
    return tuple(int(p) for p in text.split(".")[:2] if p.isdigit())


def _mlx_load_hint(macos: str, mlx: str | None) -> str | None:
    """A known cause to name when the engine fails to import on this machine.

    MLX 0.32's macOS 26 build — the ``mlx-metal`` wheel pip installs on any
    macOS 26 release — is compiled for macOS 26.2 (MLX 0.31's for 26.0).  The
    OS checks it guards with that version are folded to true, so on 26.0 and
    26.1 it assumes newer features: an M5-class GPU gets kernels the OS
    predates.  That usually fails later, at runtime, rather than at import;
    when the import itself fails on those releases, this build is still the
    first thing to rule out.

    Parameters
    ----------
    macos : str
        The running macOS version, as ``platform.mac_ver()[0]`` gives it.
    mlx : str or None
        The installed MLX version, or ``None`` when MLX is not installed.

    Returns
    -------
    str or None
        The hint to append to the import error, or ``None`` when no known
        cause applies.
    """
    if mlx is None:
        return None
    running = _version(macos)
    if not running or running[0] != 26 or running >= (26, 2):
        return None
    if _version(mlx) < (0, 32):
        return None
    return (
        f"MLX {mlx}'s macOS 26 build is compiled for macOS 26.2 or later, and "
        f"this is macOS {macos}. Update macOS, or install the previous MLX: "
        "pip install 'mlx<0.32'."
    )


try:
    from lucid._C import engine as _C_engine  # noqa: F401
except ImportError as exc:
    import importlib.metadata
    import platform

    try:
        _mlx: str | None = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        _mlx = None
    _hint = _mlx_load_hint(platform.mac_ver()[0], _mlx)
    if _hint is None:
        raise
    raise ImportError(f"{exc}\n\n{_hint}") from exc
