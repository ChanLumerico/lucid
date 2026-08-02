"""Terminal rendering for the audit CLI — standard library only.

No third-party console package.  An audit tool has to run in the one
situation where the environment is least trustworthy, so it carries no
dependency it does not need; everything here is ANSI escapes and
``shutil.get_terminal_size``.

Degrades on purpose.  When stdout is not a TTY, or ``NO_COLOR`` is set,
or ``TERM=dumb``, colour and the in-place live region are dropped and the
same information is emitted as plain sequential lines — so piping to a
file or running under CI produces a readable log rather than a screenful
of escape codes.
"""

import os
import shutil
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

# ── palette ──────────────────────────────────────────────────────────────────
# 256-colour indices, chosen to stay legible on both light and dark
# backgrounds (nothing below index 240, no pure blue on black).
_C = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "dim": "\x1b[2m",
    "red": "\x1b[38;5;203m",
    "green": "\x1b[38;5;114m",
    "yellow": "\x1b[38;5;221m",
    "blue": "\x1b[38;5;110m",
    "magenta": "\x1b[38;5;176m",
    "cyan": "\x1b[38;5;115m",
    "grey": "\x1b[38;5;245m",
    "white": "\x1b[38;5;253m",
}

_BLOCKS = " ▏▎▍▌▋▊▉█"
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def supports_colour(force: bool | None = None) -> bool:
    """Whether to emit escape sequences at all."""
    if force is not None:
        return force
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") in ("dumb", ""):
        return False
    return sys.stdout.isatty()


class Console:
    """Everything the audit prints goes through one of these.

    Parameters
    ----------
    colour : bool, optional
        Force colour on or off.  ``None`` auto-detects.
    width : int, optional
        Force a terminal width.  ``None`` measures it.
    quiet : bool, default=False
        Suppress everything except the final summary and findings.
    """

    def __init__(
        self,
        colour: bool | None = None,
        width: int | None = None,
        quiet: bool = False,
    ) -> None:
        self.colour = supports_colour(colour)
        self.live = self.colour and sys.stdout.isatty()
        self._width = width
        self.quiet = quiet
        self._live_lines = 0

    @property
    def width(self) -> int:
        """int: Usable terminal width, clamped to something readable."""
        if self._width is not None:
            return self._width
        return max(60, min(shutil.get_terminal_size((100, 24)).columns, 160))

    # ── primitives ───────────────────────────────────────────────────────────

    def paint(self, text: str, *styles: str) -> str:
        """Wrap ``text`` in the named styles, or return it unchanged."""
        if not self.colour or not styles:
            return text
        return "".join(_C[s] for s in styles if s in _C) + text + _C["reset"]

    def write(self, text: str = "") -> None:
        if self.quiet:
            return
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def always(self, text: str = "") -> None:
        """Write even in quiet mode — summaries and findings."""
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    # ── structure ────────────────────────────────────────────────────────────

    def banner(self, title: str, subtitle: str = "") -> None:
        """The block at the top of a run."""
        if self.quiet:
            return
        w = self.width
        self.write()
        self.write(self.paint("╭" + "─" * (w - 2) + "╮", "grey"))
        pad = w - 4 - len(title)
        self.write(
            self.paint("│ ", "grey")
            + self.paint(title, "bold", "cyan")
            + " " * max(pad, 0)
            + self.paint(" │", "grey")
        )
        if subtitle:
            pad = w - 4 - len(subtitle)
            self.write(
                self.paint("│ ", "grey")
                + self.paint(subtitle, "grey")
                + " " * max(pad, 0)
                + self.paint(" │", "grey")
            )
        self.write(self.paint("╰" + "─" * (w - 2) + "╯", "grey"))

    def rule(self, title: str = "", style: str = "grey") -> None:
        if self.quiet:
            return
        w = self.width
        if not title:
            self.write(self.paint("─" * w, style))
            return
        left = "── " + title + " "
        self.write(self.paint(left + "─" * max(w - len(left), 0), style))

    def kv(self, key: str, value: str, key_width: int = 22) -> None:
        if self.quiet:
            return
        self.write(
            "  " + self.paint(key.ljust(key_width), "grey") + self.paint(value, "white")
        )

    def table(
        self,
        headers: "Sequence[str]",
        rows: "Sequence[Sequence[str]]",
        aligns: "Sequence[str] | None" = None,
        always: bool = False,
    ) -> None:
        """A box-drawn table sized to its contents.

        ``aligns`` is one of ``"l"`` / ``"r"`` / ``"c"`` per column.
        """
        emit = self.always if always else self.write
        if self.quiet and not always:
            return
        if not rows:
            return
        cols = len(headers)
        aligns = list(aligns or ["l"] * cols)
        widths = [len(_strip(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row[:cols]):
                widths[i] = max(widths[i], len(_strip(str(cell))))

        def fit(cell: str, i: int) -> str:
            raw = _strip(cell)
            pad = widths[i] - len(raw)
            if aligns[i] == "r":
                return " " * pad + cell
            if aligns[i] == "c":
                half = pad // 2
                return " " * half + cell + " " * (pad - half)
            return cell + " " * pad

        bar = "─"
        emit(
            self.paint(
                "  ┌" + "┬".join(bar * (w + 2) for w in widths) + "┐",
                "grey",
            )
        )
        emit(
            self.paint("  │ ", "grey")
            + self.paint(" │ ", "grey").join(
                self.paint(fit(h, i), "bold") for i, h in enumerate(headers)
            )
            + self.paint(" │", "grey")
        )
        emit(
            self.paint(
                "  ├" + "┼".join(bar * (w + 2) for w in widths) + "┤",
                "grey",
            )
        )
        for row in rows:
            cells = [fit(str(c), i) for i, c in enumerate(row[:cols])]
            emit(
                self.paint("  │ ", "grey")
                + self.paint(" │ ", "grey").join(cells)
                + self.paint(" │", "grey")
            )
        emit(
            self.paint(
                "  └" + "┴".join(bar * (w + 2) for w in widths) + "┘",
                "grey",
            )
        )

    # ── progress ─────────────────────────────────────────────────────────────

    def bar(self, done: int, total: int, width: int = 28) -> str:
        """A sub-cell-resolution progress bar."""
        if total <= 0:
            return " " * width
        frac = max(0.0, min(1.0, done / total))
        exact = frac * width
        full = int(exact)
        rest = exact - full
        cell = _BLOCKS[int(rest * (len(_BLOCKS) - 1))] if full < width else ""
        body = "█" * full + cell
        return body + " " * max(width - len(_strip(body)), 0)

    def spinner(self, tick: int) -> str:
        return _SPINNER[tick % len(_SPINNER)]

    def live_block(self, lines: "Sequence[str]") -> None:
        """Rewrite an in-place region.

        Falls back to nothing when the terminal cannot move the cursor —
        the caller is expected to also emit real lines for the log.
        """
        if not self.live or self.quiet:
            return
        out = []
        if self._live_lines:
            out.append(f"\x1b[{self._live_lines}A")
        for line in lines:
            out.append("\x1b[2K" + line[: self.width] + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._live_lines = len(lines)

    def live_done(self) -> None:
        """Leave the live region in place and stop tracking it."""
        self._live_lines = 0


def _strip(text: str) -> str:
    """Length of ``text`` as displayed, ignoring escape sequences."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "\x1b":
            j = text.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


class Timer:
    """Elapsed / rate / ETA for the progress line."""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def eta(self, done: int, total: int) -> str:
        if done <= 0 or total <= 0:
            return "--:--"
        remaining = self.elapsed / done * (total - done)
        return fmt_duration(remaining)

    def __str__(self) -> str:
        return fmt_duration(self.elapsed)


def fmt_duration(seconds: float) -> str:
    """``m:ss`` below an hour, ``h:mm:ss`` above."""
    seconds = max(0.0, seconds)
    if seconds < 3600:
        return f"{int(seconds // 60):d}:{int(seconds % 60):02d}"
    return f"{int(seconds // 3600):d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def iter_ticks() -> "Iterator[int]":
    """An endless counter for spinner frames."""
    tick = 0
    while True:
        yield tick
        tick += 1


class Suppress:
    """Silence stdout/stderr while probing.

    Ops under audit print warnings, and a survey that emits one line per
    op would bury its own progress display.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._devnull: object | None = None
        self._saved: tuple[object, object] | None = None

    def __enter__(self) -> "Suppress":
        if not self.enabled:
            return self
        self._devnull = open(os.devnull, "w")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = self._devnull  # type: ignore[assignment]
        sys.stderr = self._devnull  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        if not self.enabled or self._saved is None:
            return
        sys.stdout, sys.stderr = self._saved  # type: ignore[assignment]
        if self._devnull is not None:
            self._devnull.close()  # type: ignore[attr-defined]
        self._devnull = None
        self._saved = None


__all__ = [
    "Console",
    "Suppress",
    "Timer",
    "fmt_duration",
    "iter_ticks",
    "supports_colour",
]
