import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lucid import models
from lucid.weights import _family_of


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _upload_to_hf(
    *,
    repo_id: str,
    revision: str,
    local_path: Path,
    remote_path: str,
    private: bool,
    token: str | None,
    commit_message: str,
) -> None:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required. Install with `pip install huggingface_hub`."
        ) from e

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Upload safetensors to HF and register entry in lucid/weights/registry.json."
        )
    )
    p.add_argument("model_key", type=str)
    p.add_argument("path", type=str, help="Local .safetensors path")
    p.add_argument("--repo", type=str, required=True, help="HF repo id: owner/repo")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--tag", type=str, default="DEFAULT")
    p.add_argument("--default", action="store_true")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--enum-name", type=str, default=None)
    p.add_argument(
        "--remote-dir",
        type=str,
        default=None,
        help="Remote directory under repo root (e.g. weights/bert).",
    )
    p.add_argument("--input-size", type=str, default=None, help="e.g. 3,224,224")
    p.add_argument(
        "--registry",
        type=str,
        default="lucid/weights/registry.json",
    )
    p.add_argument("--token", type=str, default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF upload and only update local registry.",
    )
    args = p.parse_args()

    model_key = args.model_key
    weights_path = Path(args.path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    if weights_path.suffix != ".safetensors":
        raise ValueError("path must point to a .safetensors file")

    family = _family_of(model_key)
    if not family:
        raise KeyError(f"Cannot infer family from model key: {model_key}")
    filename = weights_path.name
    if args.remote_dir:
        remote_path = f"{args.remote_dir.strip('/').rstrip('/')}/{filename}"
    else:
        family_slug = family.lower()
        remote_path = f"weights/{family_slug}/{filename}"

    if not args.no_upload:
        _upload_to_hf(
            repo_id=args.repo,
            revision=args.revision,
            local_path=weights_path,
            remote_path=remote_path,
            private=args.private,
            token=args.token,
            commit_message=f"Add {model_key}:{args.tag} weights",
        )
        print(
            f"[upload] https://huggingface.co/{args.repo}/blob/{args.revision}/{remote_path}"
        )

    h = _sha256(weights_path)
    reg_path = Path(args.registry).resolve()
    reg = _load_json(reg_path)

    meta: dict[str, Any] = {
        "hf_repo": args.repo,
        "hf_revision": args.revision,
        "hf_filename": remote_path,
        "parameter_size": getattr(models, model_key)().parameter_size,
        "family": family,
    }
    if args.input_size:
        meta["input_size"] = [int(x.strip()) for x in args.input_size.split(",")]
    if args.enum_name:
        meta["enum_name"] = args.enum_name

    entry: dict[str, Any] = {
        "url": f"hf://{args.repo}@{args.revision}/{remote_path}",
        "sha256": h,
        "meta": meta,
    }
    if args.dataset:
        entry["dataset"] = args.dataset

    reg.setdefault(model_key, {})[args.tag] = entry
    if args.default:
        reg[model_key]["DEFAULT"] = dict(entry)

    _save_json(reg_path, reg)
    print(f"[registry] updated {reg_path}")
    print(json.dumps({model_key: {args.tag: entry}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
