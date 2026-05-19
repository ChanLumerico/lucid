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
    r"""Manifest entry describing one pretrained checkpoint.

    Each family's ``_pretrained.py`` constructs one :class:`PretrainedEntry`
    per named variant (e.g. ``resnet_50``, ``resnet_101``).  The entry
    bundles the download URL, an integrity hash, the architectural
    config the weights correspond to, and human-readable metadata.

    Attributes
    ----------
    url : str
        HTTP(S) URL the weights live at.  Lucid hosts assets on GitHub
        Releases but no provider-specific knowledge is encoded here.
    sha256 : str
        Hex-encoded SHA-256 digest of the file at ``url``.  Verified
        after download and against the cached copy on every reuse — a
        mismatch triggers re-download (the cached copy is presumed
        corrupt).
    config : ModelConfig
        Architectural config the weights match.  Validated against the
        receiving model's config (``model_type`` must match) before
        loading, so mis-pairings (e.g. a ResNet-101 entry on a ResNet-50
        model) are caught early.
    num_params : int, optional, default=0
        Parameter count of the checkpoint, for user-facing reporting.
    description : str, optional, default=""
        Free-form notes — training data, accuracy on a standard
        benchmark, source paper.

    Notes
    -----
    The dataclass is ``frozen=True`` so entries can be shared by
    reference without aliasing concerns.

    Examples
    --------
    >>> entry = PretrainedEntry(
    ...     url="https://github.com/.../resnet_50.lucid",
    ...     sha256="e1b2...",
    ...     config=ResNetConfig(depth=50),
    ...     num_params=25_557_032,
    ...     description="ImageNet-1k, 76.1% top-1",
    ... )
    """

    url: str
    sha256: str
    config: ModelConfig
    num_params: int = 0
    description: str = ""


def _cache_dir() -> Path:
    r"""Resolve the cache root, honouring ``LUCID_HOME`` if set.

    Returns
    -------
    pathlib.Path
        Cache root for model weights — ``$LUCID_HOME/models`` when the
        environment variable is set, ``~/.cache/lucid/models`` otherwise.

    Notes
    -----
    The function does not create the directory; callers
    (:func:`download`) ``mkdir(parents=True, exist_ok=True)`` lazily on
    first use.
    """
    base = os.environ.get("LUCID_HOME")
    root = Path(base) if base is not None else Path.home() / ".cache" / "lucid"
    return root / "models"


def _verify_sha256(path: Path, expected: str) -> None:
    r"""Raise :class:`RuntimeError` if the file at ``path`` does not hash to
    ``expected``.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.
    expected : str
        Hex-encoded expected SHA-256 digest.

    Raises
    ------
    RuntimeError
        On hash mismatch.  The message includes both expected and
        observed digests so callers can debug corruption / supply-chain
        issues.

    Notes
    -----
    Streams the file in 1 MiB chunks so very large checkpoints don't
    require loading the full payload into memory.
    """
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
    r"""Download ``url`` to the local cache, verify SHA-256, return the path.

    The function is idempotent — when the file already exists in cache
    and verifies, it is returned immediately without re-fetching.  A
    cached file that *fails* verification is treated as corrupt and
    re-downloaded.

    Parameters
    ----------
    url : str
        Remote location of the weights blob.
    sha256 : str
        Hex-encoded expected SHA-256 digest.  Validated after download
        and on every cache hit.
    name : str, keyword-only
        Short identifier (typically the registered model name).  Used to
        compose the cache subdirectory so different models don't share
        on-disk paths.

    Returns
    -------
    pathlib.Path
        Local file path of the verified checkpoint.

    Raises
    ------
    RuntimeError
        If the freshly downloaded file fails SHA-256 verification.
    OSError / urllib.error.URLError
        From the underlying ``urllib.request.urlopen`` call when the
        network is unreachable or the URL is invalid.

    Notes
    -----
    Cache layout::

        {cache_root}/{name}/{sha256[:8]}/weights.lucid

    Download is atomic: bytes go to a temp file in the cache directory,
    SHA is verified, then the temp file is renamed into place.  On
    failure (network error, hash mismatch) the temp file is removed but
    the cache directory is preserved for the next attempt.

    Examples
    --------
    >>> path = download(
    ...     url="https://github.com/.../resnet_50.lucid",
    ...     sha256="e1b2...",
    ...     name="resnet_50",
    ... )
    >>> path.exists()
    True
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
    r"""Download weights described by ``entry`` and load them into ``model``.

    Glues together :class:`PretrainedEntry`, :func:`download`, and
    :meth:`PretrainedModel.load_state_dict`.  Family factories call this
    from their ``pretrained=True`` code path.

    Parameters
    ----------
    model : PretrainedModel
        Destination model — its ``config.model_type`` must equal
        ``entry.config.model_type``.
    entry : PretrainedEntry
        Manifest entry carrying the URL, hash, and config.
    name : str, keyword-only
        Short identifier passed through to :func:`download` so weights
        for different models live in separate cache subdirectories.
    strict : bool, optional, keyword-only, default=True
        Forwarded to :meth:`PretrainedModel.load_state_dict`.  When
        ``False``, missing / extra keys are tolerated (useful when the
        target model has a different head shape from the checkpoint).

    Raises
    ------
    TypeError
        If ``entry.config.model_type`` does not match
        ``model.config.model_type`` — guards against accidentally pairing
        e.g. a ResNet-50 entry with a ResNet-18 model.
    RuntimeError
        On SHA-256 verification failure during download.

    Notes
    -----
    Downloads land in the per-name SHA-prefixed cache (see
    :func:`download`).  Subsequent loads of the same entry are
    effectively zero-cost (a single SHA verification of the local file).

    Examples
    --------
    >>> from lucid.models import PretrainedEntry, load_from_pretrained_entry
    >>> entry = PretrainedEntry(url=..., sha256=..., config=cfg)
    >>> model = ResNet(cfg)
    >>> load_from_pretrained_entry(model, entry, name="resnet_50")
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
