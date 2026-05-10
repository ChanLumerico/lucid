"""Pretrained-weight download from URLs with SHA256 verification + local cache.

Lucid hosts weights as GitHub Release assets; this module knows nothing about
that — it just follows a URL.  Callers (each family's ``pretrained.py``)
supply ``url + sha256`` via :class:`PretrainedEntry`.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lucid as _lucid

if TYPE_CHECKING:
    from lucid.models._base import ModelConfig, PretrainedModel


@dataclass(frozen=True)
class PretrainedEntry:
    """Manifest entry for one pretrained checkpoint."""

    url: str
    sha256: str
    config: ModelConfig
    num_params: int = 0
    description: str = ""


def _cache_dir() -> Path:
    """Resolve the cache root, honoring ``LUCID_HOME`` if set."""
    base = os.environ.get("LUCID_HOME")
    root = Path(base) if base is not None else Path.home() / ".cache" / "lucid"
    return root / "models"


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path.name}: expected {expected}, got {got}"
        )


def download(url: str, sha256: str, *, name: str) -> Path:
    """Download ``url`` to the cache, verify SHA256, return local file path.

    Cache layout: ``{cache_root}/{name}/{sha256[:8]}/weights.lucid``.
    Re-download is skipped when the cached file already verifies.
    """
    cache_root = _cache_dir() / name / sha256[:8]
    cache_root.mkdir(parents=True, exist_ok=True)
    final = cache_root / "weights.lucid"

    if final.exists():
        try:
            _verify_sha256(final, sha256)
            return final
        except RuntimeError:
            # Stale / corrupted — drop and re-download.
            final.unlink()

    fd, tmp_name = tempfile.mkstemp(dir=cache_root, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        _verify_sha256(tmp_path, sha256)
        tmp_path.replace(final)
    except BaseException:
        # Any failure: clean the temp file but keep the cache dir
        # (it's reused on the next attempt).
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return final


def load_from_pretrained_entry(
    model: PretrainedModel,
    entry: PretrainedEntry,
    *,
    name: str,
    strict: bool = True,
) -> None:
    """Download weights via *entry* and load them into *model*.

    Validates that *entry.config* and *model.config* share the same
    ``model_type`` so mis-matched factory / entry pairs are caught early
    (e.g., passing a ResNet-50 entry to a ResNet-18 model).
    """
    if entry.config.model_type != model.config.model_type:
        raise TypeError(
            f"load_from_pretrained_entry: entry config model_type "
            f"{entry.config.model_type!r} does not match model config "
            f"model_type {model.config.model_type!r}. "
            f"Check that the correct PretrainedEntry is paired with this model."
        )
    weights_path = download(entry.url, entry.sha256, name=name)
    sd = _lucid.load(str(weights_path), weights_only=True)
    if not isinstance(sd, dict):
        raise TypeError(
            f"weights file did not contain a state_dict, got {type(sd).__name__}"
        )
    model.load_state_dict(sd, strict=strict)
