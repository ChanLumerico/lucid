"""Conversion engine: source state_dict → verified Lucid safetensors.

Architecture-agnostic core.  Concrete architectures (e.g.
:mod:`tools.convert_weights.resnet`) subclass :class:`Architecture` to
supply three things: how to fetch the upstream weights, how to build the
matching empty Lucid model, and how to remap each key.  Everything else
— key 1:1 verification, shape checks, a real ``load_state_dict`` smoke
load, safetensors writing with embedded metadata, and ``config.json`` /
``README.md`` rendering — is handled here.
"""

import abc
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import lucid
import lucid.serialization as _serial
from lucid._factories.converters import from_numpy
from lucid.nn import Module

from tools.convert_weights._templates import render_config_json, render_model_card


@dataclass
class ConversionSpec:
    """Static metadata describing one checkpoint conversion.

    Carries everything the templates + safetensors header need that is
    not derivable from the weights themselves.
    """

    model_name: str
    """Lucid factory the weights load into, e.g. ``"resnet_18_cls"``."""
    architecture: str
    """Canonical short name used in URLs / cards, e.g. ``"resnet_18"``."""
    repo_id: str
    """Hub repo, e.g. ``"lucid-dl/resnet-18"``."""
    tag: str
    """Variant tag, e.g. ``"IMAGENET1K_V1"``."""
    task: str
    """HF pipeline tag, e.g. ``"image-classification"``."""
    model_type: str
    """Lucid config ``model_type``, e.g. ``"resnet"``."""
    source: str
    """Provenance string, e.g. ``"torchvision/ResNet18_Weights.IMAGENET1K_V1"``."""
    license: str
    """SPDX-ish license id inherited from the source weights."""
    num_classes: int
    """Output-class count of the head."""
    config: dict[str, object]
    """Lucid config dataclass dump (architecture hyperparameters)."""
    preprocessing: dict[str, object]
    """Inference transform spec (resize/crop/mean/std/interpolation)."""
    citation: str
    """Paper citation (BibTeX or free text) for the model card."""
    title: str
    """Human title, e.g. ``"ResNet-18"``."""
    paper_url: str = ""
    """Optional paper link rendered in the card header."""
    categories: list[str] = field(default_factory=list)
    """Class label names (e.g. ImageNet-1k), used to build id2label."""
    meta: dict[str, object] = field(default_factory=dict)
    """Metrics + recipe + flops, merged into config.json + card."""


class Architecture(abc.ABC):
    """Per-architecture conversion recipe.

    Subclasses bind a concrete upstream checkpoint (via ``__init__``
    args like ``tag``) and implement the three abstract hooks.
    """

    @abc.abstractmethod
    def source_state_dict(self) -> dict[str, object]:
        """Return the upstream ``state_dict`` as ``{name: numpy.ndarray}``."""
        raise NotImplementedError

    @abc.abstractmethod
    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        raise NotImplementedError

    @abc.abstractmethod
    def map_key(self, src_key: str) -> str | None:
        """Map an upstream key to its Lucid name (``None`` drops the key)."""
        raise NotImplementedError

    def transform_value(self, src_key: str, arr: object) -> object:
        """Optional hook — reshape / cast the raw upstream array per key.

        Default identity.  Override on subclasses that need to massage a
        few specific tensors before they land in Lucid (e.g. ConvNeXt's
        ``layer_scale`` ships as ``(C, 1, 1)`` for use in NCHW
        broadcasting but Lucid stores it as ``(C,)`` for an explicit
        elementwise multiply).
        """
        return arr

    @abc.abstractmethod
    def spec(self) -> ConversionSpec:
        """Return the static :class:`ConversionSpec` for this checkpoint."""
        raise NotImplementedError


def convert(arch: Architecture) -> tuple[dict[str, object], ConversionSpec]:
    """Remap + verify an architecture's weights into a Lucid state dict.

    Performs the full safety gauntlet: key remap, 1:1 key-set check
    against a freshly-built Lucid model, per-key shape check, and a real
    ``load_state_dict(strict=True)`` smoke load.  Any discrepancy raises
    before anything is written.

    Parameters
    ----------
    arch : Architecture
        The conversion recipe.

    Returns
    -------
    (dict, ConversionSpec)
        The Lucid state dict (``{lucid_key: Tensor}``) and the spec.

    Raises
    ------
    RuntimeError
        On missing / extra keys or a shape mismatch.
    """
    src = arch.source_state_dict()
    model = arch.target_model()
    target_sd = model.state_dict()
    target_keys = set(target_sd.keys())

    out: dict[str, object] = {}
    for src_key, arr in src.items():
        lucid_key = arch.map_key(src_key)
        if lucid_key is None:
            continue
        if lucid_key in out:
            raise RuntimeError(
                f"convert: key collision — two source keys map to {lucid_key!r}"
            )
        out[lucid_key] = from_numpy(arch.transform_value(src_key, arr))

    got = set(out)
    missing = sorted(target_keys - got)
    extra = sorted(got - target_keys)
    if missing or extra:
        raise RuntimeError(
            f"convert: key-set mismatch for {arch.spec().model_name}.\n"
            f"  missing ({len(missing)}): {missing[:10]}\n"
            f"  extra   ({len(extra)}): {extra[:10]}"
        )

    for key in sorted(got):
        want = tuple(int(d) for d in target_sd[key].shape)
        have = tuple(int(d) for d in out[key].shape)
        if want != have:
            raise RuntimeError(
                f"convert: shape mismatch at {key!r}: target {want}, got {have}"
            )

    # Real load — the ultimate correctness gate.
    model.load_state_dict(out, strict=True)
    return out, arch.spec()


def _sha256_of(path: Path) -> str:
    """Stream-hash a file in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(
    state_dict: dict[str, object],
    spec: ConversionSpec,
    out_root: str,
) -> Path:
    """Write ``model.safetensors`` + ``config.json`` + ``README.md``.

    Layout::

        {out_root}/{repo_id}/{tag}/model.safetensors
        {out_root}/{repo_id}/{tag}/config.json
        {out_root}/{repo_id}/README.md            (repo-level card)

    The safetensors file embeds a flat ``__metadata__`` header (format,
    architecture, tag, source, license, sha-less scalar facts); the
    full rich ``meta`` is JSON-encoded into ``config.json``.  ``sha256``
    is computed post-write and injected into both the card and the
    returned spec's meta.

    Parameters
    ----------
    state_dict : dict
        Verified Lucid state dict from :func:`convert`.
    spec : ConversionSpec
        Static metadata.
    out_root : str
        Output root directory (e.g. ``"./_converted"``).

    Returns
    -------
    pathlib.Path
        The per-tag directory that was written.
    """
    repo_dir = Path(out_root) / spec.repo_id
    tag_dir = repo_dir / spec.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    st_path = tag_dir / "model.safetensors"
    header = {
        "format": "lucid",
        "lucid_version": str(lucid.__version__),
        "architecture": spec.architecture,
        "model_type": spec.model_type,
        "tag": spec.tag,
        "source": spec.source,
        "license": spec.license,
        "num_classes": str(spec.num_classes),
    }
    _serial.save_safetensors(state_dict, str(st_path), metadata=header)

    sha = _sha256_of(st_path)
    file_size_mb = round(st_path.stat().st_size / (1024 * 1024), 2)
    spec.meta = {**spec.meta, "sha256": sha, "file_size_mb": file_size_mb}

    (tag_dir / "config.json").write_text(
        render_config_json(spec), encoding="utf-8"
    )
    (repo_dir / "README.md").write_text(
        render_model_card(spec), encoding="utf-8"
    )
    return tag_dir


def upload(tag_dir: Path, spec: ConversionSpec) -> str:
    """Upload a converted checkpoint folder to the Hugging Face Hub.

    Pushes the per-tag folder into ``spec.repo_id`` under the tag
    subfolder, and the repo-level ``README.md`` to the repo root.
    Requires ``huggingface-cli login`` or ``HF_TOKEN`` in the env.

    Parameters
    ----------
    tag_dir : pathlib.Path
        The per-tag directory produced by :func:`write`.
    spec : ConversionSpec
        Static metadata (provides ``repo_id`` + ``tag``).

    Returns
    -------
    str
        The Hub URL of the updated repo.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(spec.repo_id, repo_type="model", exist_ok=True)
    api = HfApi()
    # Tag subfolder (weights + per-tag config).
    api.upload_folder(
        repo_id=spec.repo_id,
        folder_path=str(tag_dir),
        path_in_repo=spec.tag,
        commit_message=f"Add {spec.tag} weights",
    )
    # Repo-level card.
    readme = tag_dir.parent / "README.md"
    if readme.is_file():
        api.upload_file(
            repo_id=spec.repo_id,
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            commit_message="Update model card",
        )
    return f"https://huggingface.co/{spec.repo_id}"


# Registry of convertible architectures, populated by the per-arch modules.
_ARCH_BUILDERS: dict[str, Callable[[str], Architecture]] = {}


def register_arch(model_name: str) -> Callable[
    [Callable[[str], Architecture]], Callable[[str], Architecture]
]:
    """Register an ``Architecture`` builder under a model name for the CLI."""

    def _decorator(
        builder: Callable[[str], Architecture],
    ) -> Callable[[str], Architecture]:
        _ARCH_BUILDERS[model_name] = builder
        return builder

    return _decorator


def get_arch(model_name: str, tag: str) -> Architecture:
    """Look up + instantiate a registered architecture for ``tag``."""
    if model_name not in _ARCH_BUILDERS:
        raise KeyError(
            f"get_arch: no converter registered for {model_name!r}. "
            f"Known: {sorted(_ARCH_BUILDERS)}"
        )
    return _ARCH_BUILDERS[model_name](tag)
