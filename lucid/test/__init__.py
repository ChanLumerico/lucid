try:
    import torch
except ModuleNotFoundError:
    torch = None
    raise ModuleNotFoundError("'torch' is required to perform pytest on Lucid.")

del torch

from collections.abc import Sequence
from typing import Any
from pathlib import Path
import sys

from pytest import main as _pytest_main


__all__ = ["run_tests"]


def run_tests(
    paths: str | Path | Sequence[str | Path] = "lucid/test",
    *,
    raise_on_fail: bool = False,
    show_progress: bool = True,
    concise_failures: bool = True,
    single_line: bool = True,
) -> int:
    test_paths = _normalize_test_paths(paths)
    return _run_pytest_with_progress(
        test_paths,
        show_progress=show_progress,
        raise_on_fail=raise_on_fail,
        concise_failures=concise_failures,
        single_line=single_line,
    )


def _normalize_test_paths(paths: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(paths, (str, Path)):
        return [str(paths)]
    return [str(p) for p in paths]


def _run_pytest_with_progress(
    paths: list[str],
    *,
    show_progress: bool,
    raise_on_fail: bool,
    concise_failures: bool,
    single_line: bool,
) -> int:
    _ = single_line
    if not show_progress:
        return _run_pytest(paths, raise_on_fail=raise_on_fail)

    try:
        from tqdm import tqdm
    except Exception as exc:
        raise RuntimeError("tqdm is required for progress display.") from exc

    plugin = _TqdmProgressPlugin(tqdm, concise_failures=concise_failures)

    exit_code = _pytest_main(
        ["-q", "-p", "no:terminalreporter", *paths], plugins=[plugin]
    )
    if raise_on_fail and exit_code != 0:
        raise AssertionError(f"pytest exited with code: {exit_code}")
    return exit_code


from .core import *


def _run_pytest(paths: list[str], *, raise_on_fail: bool = False) -> int:
    exit_code = _pytest_main(["-q", *paths])
    if raise_on_fail and exit_code != 0:
        raise AssertionError(f"pytest exited with code: {exit_code}")
    return exit_code


class _PytestProgressPluginBase:
    def __init__(self, concise_failures: bool = True) -> None:
        self._concise_failures: bool = concise_failures
        self._failure_infos: list[tuple[str, str]] = []
        self._passed: int = 0
        self._failed: int = 0
        self._counted: set[str] = set()
        self._total: int = 0
        self._label_chars: int = 42

    @staticmethod
    def _colorize(text: str, color_code: str = "") -> str:
        if not sys.stderr.isatty():
            return text
        return f"\033[{color_code}m{text}\033[0m"

    def _green(self, text: str) -> str:
        return self._colorize(text, "92")

    def _red(self, text: str) -> str:
        return self._colorize(text, "91")

    def _yellow(self, text: str) -> str:
        return self._colorize(text, "93")

    def _cyan(self, text: str) -> str:
        return self._colorize(text, "96")

    def _bold(self, text: str) -> str:
        return self._colorize(text, "1")

    def _get_short_label(self, nodeid: str | None) -> str:
        label = nodeid or "processing"
        max_label_len = max(8, self._label_chars)
        short_label = label.split("::")[-1]
        if len(short_label) > max_label_len:
            short_label = f"{short_label[: max_label_len - 3]}..."
        return short_label

    def _record_case_result(self, report: Any) -> None:
        if report.passed:
            self._passed += 1
        if report.failed:
            self._failed += 1

    def _build_summary_postfix(self, nodeid: str | None) -> str:
        return f"{self._get_short_label(nodeid)} | passed={self._passed} failed={self._failed}"

    def _add_failure(self, report: Any) -> None:
        if not self._concise_failures:
            return
        if report.outcome not in {"failed", "error"}:
            return
        raw = str(report.longrepr).splitlines()
        message = raw[-1] if raw else "failed"
        self._failure_infos.append((report.nodeid, message))

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        print(file=sys.stderr)
        if self._failed == 0:
            summary = self._green("ALL PASSED")
        else:
            summary = self._red(f"FAILED ({self._failed})")
        print(
            self._bold(
                f"{summary} | total={self._total}, passed={self._passed}, failed={self._failed}"
            ),
            file=sys.stderr,
        )

        if self._concise_failures and self._failure_infos:
            print(file=sys.stderr)
            print(self._yellow("Failing cases:"), file=sys.stderr)
            for idx, (nodeid, message) in enumerate(self._failure_infos, start=1):
                print(f" {idx}. {self._cyan(nodeid)}", file=sys.stderr)
                print(f"    {message}", file=sys.stderr)


class _TqdmProgressPlugin(_PytestProgressPluginBase):
    def __init__(self, tqdm_mod: Any, concise_failures: bool = True) -> None:
        super().__init__(concise_failures=concise_failures)
        self._tqdm_mod = tqdm_mod
        self._bar: Any | None = None

    def pytest_collection_modifyitems(
        self, session: Any, config: Any, items: Sequence[Any]
    ) -> None:
        self._total = len(items)
        self._bar = self._tqdm_mod(
            total=len(items),
            desc="Running tests",
            file=sys.stderr,
            unit="case",
            leave=False,
        )

    def _set_postfix(self, nodeid: str | None = None) -> None:
        if self._bar is None:
            return
        self._bar.set_postfix_str(self._build_summary_postfix(nodeid), refresh=False)

    def pytest_runtest_logreport(self, report: Any) -> None:
        if self._bar is None:
            return
        if report.when != "call":
            return
        if report.nodeid in self._counted:
            return

        self._counted.add(report.nodeid)
        self._record_case_result(report)
        self._set_postfix(report.nodeid)
        self._add_failure(report)
        self._bar.update(1)

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        if self._bar is not None:
            self._bar.close()
        super().pytest_sessionfinish(session=session, exitstatus=exitstatus)
