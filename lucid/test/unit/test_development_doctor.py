"""A broken native installation must still explain how to recover."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import doctor


def test_stale_engine_import_gives_rebuild_command() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, types
engine = types.ModuleType('lucid._C.engine')
engine.ABI_VERSION = -1
sys.modules['lucid._C.engine'] = engine
import lucid
""",
        ],
        cwd=doctor.ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "got -1" in result.stderr
    assert doctor.REBUILD in result.stderr
    assert "tools.doctor" in result.stderr


def test_doctor_detects_source_contract_drift(tmp_path: Path) -> None:
    (tmp_path / "lucid/_C").mkdir(parents=True)
    (tmp_path / "lucid/version.py").write_text("_EXPECTED_ABI: int = 12\n")
    (tmp_path / "lucid/_C/version.h").write_text("#define LUCID_ABI_VERSION 11\n")
    assert doctor.source_abi(tmp_path) == (12, 11)


@pytest.mark.parametrize("returncode", [1, -6])
def test_doctor_reports_failed_or_crashed_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    returncode: int,
) -> None:
    def failed_probe(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["python"], returncode, "", "native load failed"
        )

    monkeypatch.setattr(doctor.subprocess, "run", failed_probe)
    assert doctor.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    runtime = next(check for check in report["checks"] if check["name"] == "runtime")
    assert runtime["status"] == "error"
    assert f"exit={returncode}" in runtime["detail"]
    assert doctor.REBUILD in runtime["detail"]


def test_doctor_reports_hung_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def hung_probe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("python", 0.1)

    monkeypatch.setattr(doctor.subprocess, "run", hung_probe)
    runtime = next(
        check for check in doctor.diagnose(0.1) if check["name"] == "runtime"
    )
    assert runtime["status"] == "error"
    assert "timed out" in runtime["detail"]
