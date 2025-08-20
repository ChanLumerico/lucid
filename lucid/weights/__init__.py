from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, OrderedDict

import os, json, hashlib, urllib.request, tempfile, shutil

import lucid
import lucid.nn as nn

from lucid._tensor import Tensor


_WEIGHTS_DIR: Path = Path(__file__).resolve().parent
WEIGHTS_REGISTRY_PATH: Path = _WEIGHTS_DIR / "registry.json"

CACHE_HOME = (
    Path(os.environ.get("LUCID_HOME", Path.home() / ".cache" / "lucid")) / "weights"
)
BASE_URL = os.environ.get("LUCID_WEIGHTS_BASE", "")
MIRRORS = [
    s for s in os.environ.get("LUCID_WEIGHTS_MIRRORS", "").split(",") if s.strip()
]
OFFLINE = os.environ.get("LUCID_WEIGHTS_OFFLINE", "0") == "1"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF_TOKEN = (
    os.environ.get("LUCID_HF_TOKEN")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)


@dataclass(frozen=True)
class WeightEntry:
    url: str
    sha256: str
    tag: str
    dataset: str | None = None
    meta: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


_REGISTRY = _load_json(WEIGHTS_REGISTRY_PATH)
_MODELS_META = _load_json(lucid.MODELS_REGISTRY_PATH)

_DYNAMIC: dict[str, Enum] = {}


def _canon(x: str) -> str:
    return x.lower()


def _family_of(model_key: str) -> str | None:
    if model_key in _MODELS_META:
        return _MODELS_META[model_key].get("family")
    return None


def _classname(model_key: str, family: str | None = None) -> str:
    if family is not None:
        return family + model_key[len(family) :].upper() + "_Weights"

    fam = _family_of(model_key)
    if fam:
        suf = model_key
        fam_key = fam.lower().replace(" ", "")
        suf = suf[len(fam_key) :] if suf.replace("_", "").startswith(fam_key) else suf
        suf = suf.lstrip()

        return f"{fam}{suf.upper()}_Weights" if suf else f"{fam}_Weights"

    parts = model_key.split("_")
    return "_".join(p.capitalize() for p in parts) + "_Weights"


def _make_enum(model_key: str, entries: dict[str, Any]) -> type[Enum]:
    members = {}
    for tag, info in entries.items():
        members[tag] = WeightEntry(
            url=info["url"],
            sha256=info["sha256"],
            tag=tag,
            dataset=info.get("dataset"),
            meta=info.get("meta", {}),
        )

    cl = Enum(
        _classname(model_key, family=entries["DEFAULT"]["meta"].get("family", None)),
        members,
    )
    _DYNAMIC[_canon(model_key)] = cl
    return cl


_created_names: list[str] = []
for k, entries in _REGISTRY.items():
    cl = _make_enum(k, entries)
    globals()[cl.__name__] = cl
    _created_names.append(cl.__name__)


def get_enum(model_key: str) -> type[Enum]:
    key = _canon(model_key)
    if key not in _DYNAMIC:
        raise KeyError(f"No weights registered for '{model_key}'.")
    return _DYNAMIC[key]


def resolve(weights: Enum | str, *, model_key: str | None = None) -> WeightEntry:
    if isinstance(weights, Enum):
        return weights.value
    if model_key is None:
        raise ValueError("model_key is required when resolving from string.")

    enum = get_enum(model_key)
    try:
        return getattr(enum, weights).value
    except AttributeError:
        return getattr(enum, "DEFAULT").value


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_build_url(repo_id: str, revision: str, filename: str) -> str:
    rid = repo_id.strip("/")
    rev = revision or "main"
    return f"{HF_ENDPOINT.rstrip("/")}/{rid}/resolve/{rev}/{filename}?download=true"


def _parse_hf_uri(uri: str) -> tuple[str, str, str]:
    if not uri.startswith("hf://"):
        raise ValueError("Not an hf:// URI")

    s = uri[5:].lstrip("/")
    parts = s.split("/")
    if len(parts) < 3:
        raise ValueError("HF URI must be 'hf://owner/repo[@rev]/path'")

    owner = parts[0]
    repo_and_rev = parts[1]
    file_path = "/".join(parts[2:])

    if "@" in repo_and_rev:
        repo, rev = repo_and_rev.split("@", 1)
    else:
        repo, rev = repo_and_rev, "main"

    repo_id = f"{owner}/{repo}"
    return repo_id, rev, file_path


def _hf_candidate_urls(entry: WeightEntry) -> list[str]:
    urls: list[str] = []
    try:
        if entry.url.startswith("hf://"):
            repo_id, rev, file_path = _parse_hf_uri(entry.url)
            urls.append(_hf_build_url(repo_id, rev, file_path))
    except Exception:
        pass

    m = entry.meta or {}
    repo = m.get("hf_repo")
    file = m.get("hf_filename") or m.get("hf_path")
    rev = m.get("hf_revision", "main")

    if repo and file:
        urls.append(_hf_build_url(repo, rev, file))
    return urls


def _candidate_urls(entry: WeightEntry) -> list[str]:
    urls = []
    urls.extend(_hf_candidate_urls(entry))

    if entry.url and not entry.url.startswith("hf://"):
        urls.append(entry.url)
        if BASE_URL and entry.url.startswith("http"):
            tail = "/".join(entry.url.split("/")[3:])
            urls.insert(0, BASE_URL.rstrip("/") + "/" + tail)

        for m in MIRRORS:
            m = m.strip()
            if not m:
                continue
            if entry.url.startswith("http"):
                tail = "/".join(entry.url.split("/")[3:])
                urls.append(m.rstrip("/") + "/" + tail)
            else:
                urls.append(m.rstrip("/") + "/" + entry.url.lstrip("/"))

    return urls


def _cache_path(model_key: str, entry: WeightEntry) -> Path:
    fname = f"{_canon(model_key)}--{entry.tag}--{entry.sha256[:8]}.safetensors"
    return CACHE_HOME / fname


def load_state_dict_from_url(
    model_key: str,
    entry: WeightEntry,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict:
    cache = cache_dir or CACHE_HOME
    cache.mkdir(parents=True, exist_ok=True)
    dst = _cache_path(model_key, entry)

    if not OFFLINE and (force or not dst.exists()):
        tmp_path = Path(tempfile.mkstemp(prefix="lucid-", suffix=".tmp")[1])
        last_err = None
        try:
            for u in _candidate_urls(entry):
                try:
                    req = urllib.request.Request(u)
                    if HF_TOKEN and (HF_ENDPOINT in u or "huggingface.co" in u):
                        req.add_header("Authorization", f"Bearer {HF_TOKEN}")
                    with urllib.request.urlopen(req) as r, open(tmp_path, "wb") as out:
                        shutil.copyfileobj(r, out)
                    break

                except Exception as e:
                    last_err = e
                    continue
            else:
                if last_err:
                    raise last_err

            if _hash_file(tmp_path) != entry.sha256:
                raise RuntimeError("Checksum mismatch for downloaded weights.")
            shutil.move(tmp_path, dst)

        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    elif not dst.exists():
        raise FileNotFoundError(
            "Weights not present in cache and offline mode is enabled."
        )

    if _hash_file(dst) != entry.sha256:
        dst.unlink()
        return load_state_dict_from_url(model_key, entry, cache_dir=cache, force=True)

    try:
        from safetensors.numpy import load_file
    except Exception as e:
        raise ImportError(
            "safetensors is required to save .safetensors files. "
            "Install with `pip install safetensors`."
        ) from e

    state = load_file(str(dst))
    return OrderedDict((k, v) for k, v in state.items())


def apply(model: nn.Module, weights: Enum | str, *, strict: bool = True) -> Path:
    key = getattr(model, "_alt_name", type(model).__name__).lower()
    entry = resolve(weights, model_key=key)
    state = load_state_dict_from_url(key, entry)

    device = getattr(model, "device", "cpu")
    if device == "gpu":
        cooked = {}
        for k, v in state.items():
            if hasattr(v, "shape"):
                cooked[k] = Tensor(v, device="gpu")
            else:
                cooked[k] = v
        state = cooked

    model.load_state_dict(state, strict=strict)
    return _cache_path(key, entry)


def list_available(model_key: str) -> dict[str, WeightEntry]:
    key = _canon(model_key)
    if key not in _REGISTRY:
        return {}
    enum = get_enum(key)
    return {m.name: m.value for m in enum}


__all__ = [
    "WeightEntry",
    "get_enum",
    "resolve",
    "apply",
    "list_available",
    "load_state_dict_from_url",
    *_created_names,
]
