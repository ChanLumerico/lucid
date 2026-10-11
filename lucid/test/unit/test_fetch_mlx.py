"""``tools/fetch_mlx.py`` fails with a remedy, never with a raw traceback.

It runs first in every engine build, including an sdist build on a user's
machine, so each way it can fail — no network, no Xcode tools, an
incomplete offline dist, a concurrent build — has to say what to do.
"""

import io
from pathlib import Path

import pytest

from tools import fetch_mlx


def _pin(url: str) -> fetch_mlx.Pin:
    return fetch_mlx.Pin("0.0.0", "w.whl", "0" * 64, url, "26.2")


def test_an_unreachable_url_names_the_offline_remedy(tmp_path: Path) -> None:
    url = (tmp_path / "missing.whl").as_uri()
    with pytest.raises(fetch_mlx.FetchError, match="LUCID_MLX_DIST_DIR") as info:
        fetch_mlx.fetch_wheel(_pin(url), tmp_path)
    assert url in str(info.value)
    assert not list(tmp_path.glob(".download-*")), "a failed download left a partial file"


def test_a_corrupt_download_is_refused(tmp_path: Path) -> None:
    payload = tmp_path / "payload.whl"
    payload.write_bytes(b"not the pinned wheel")
    with pytest.raises(fetch_mlx.FetchError, match="sha256"):
        fetch_mlx.fetch_wheel(_pin(payload.as_uri()), tmp_path)
    assert not (tmp_path / f"{'0' * 64}.whl").exists()


def test_a_missing_xcode_tool_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    with pytest.raises(fetch_mlx.FetchError, match="Xcode command line tools"):
        fetch_mlx._run("install_name_tool", "-id", "x", "y")


def _dist(root: Path, skip: str | None = None) -> Path:
    (root / "mlx/include").mkdir(parents=True)
    (root / "mlx/lib").mkdir()
    for name in ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib"):
        if name != skip:
            (root / "mlx/lib" / name).write_bytes(b"")
    return root


@pytest.mark.parametrize("missing", ["libmlx.dylib", "libjaccl.dylib", "mlx.metallib"])
def test_an_offline_dist_must_hold_every_runtime_file(tmp_path: Path, missing: str) -> None:
    with pytest.raises(fetch_mlx.FetchError, match=missing.replace(".", r"\.")):
        fetch_mlx._locate_dist(_dist(tmp_path, skip=missing))


def test_an_offline_dist_without_a_license_is_refused(tmp_path: Path) -> None:
    root = fetch_mlx._locate_dist(_dist(tmp_path))
    with pytest.raises(fetch_mlx.FetchError, match="LICENSE"):
        fetch_mlx._dist_license(root)
    (root.parent / "mlx_metal-0.0.0.dist-info/licenses").mkdir(parents=True)
    (root.parent / "mlx_metal-0.0.0.dist-info/licenses/LICENSE").write_text("MIT")
    assert fetch_mlx._dist_license(root).read_text() == "MIT"


def test_a_concurrent_winner_is_reused(tmp_path: Path) -> None:
    entry = tmp_path / "mlx-0.0.0-abc"
    entry.mkdir()
    (entry / "lib").mkdir()
    (entry / fetch_mlx.STAMP).write_text("winner\n")
    loser = tmp_path / ".stage-loser"
    loser.mkdir()
    assert fetch_mlx._publish(loser, entry, "loser") == entry
    assert not loser.exists()
    assert (entry / fetch_mlx.STAMP).read_text() == "winner\n"


def test_bump_keeps_the_floor_major(monkeypatch: pytest.MonkeyPatch) -> None:
    """A release that adds a macOS 27 build must not raise Lucid's floor."""
    tags = ["14_0", "15_0", "26_0", "27_0"]
    listing = [(f"mlx_metal-9.9.9-py3-none-macosx_{t}_arm64.whl", t, f"u/{t}") for t in tags]
    listing.append(("mlx_metal-9.9.9.tar.gz", "src", "u/src"))
    monkeypatch.setattr(fetch_mlx, "_pypi_wheels", lambda version: listing)
    pin = fetch_mlx._resolve("9.9.9", "26.2")
    assert pin.wheel.endswith("macosx_26_0_arm64.whl")
    assert (pin.sha256, pin.url) == ("26_0", "u/26_0")


def test_download_streams_into_the_given_file(tmp_path: Path) -> None:
    source = tmp_path / "blob"
    source.write_bytes(b"x" * 3_000_000)
    out = io.BytesIO()
    fetch_mlx._download(source.as_uri(), out)
    assert out.getvalue() == source.read_bytes()
