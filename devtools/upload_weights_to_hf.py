import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_and_path(repo: str) -> tuple[str, str]:
    if "@" in repo:
        repo_id, rev = repo.split("@", 1)
        return repo_id, rev
    return repo, "main"


def _normalize_tag(tag: str | None) -> str:
    if not tag:
        return ""
    s = tag.strip()
    if not s:
        return s
    return s.upper() if s.isascii() and s != s.upper() else s


def _infer_model_key(
    filename: str,
    family: str,
    explicit_model_key: str | None,
    explicit_dataset: str | None = None,
) -> tuple[str, str]:
    if explicit_model_key:
        if explicit_dataset:
            return explicit_model_key, explicit_dataset
        return explicit_model_key, ""

    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Cannot infer model from '{filename}'. Use --model-key explicitly."
        )

    family_l = family.lower()
    if parts[0].lower() != family_l:
        raise ValueError(
            f"filename '{filename}' does not start with family '{family}'. "
            f"Use --model-key explicitly."
        )

    rest = "_".join(parts[1:])
    if not rest:
        raise ValueError(f"filename '{filename}' has no model variant part.")

    dataset = explicit_dataset or parts[-1]
    if explicit_dataset:
        tail = explicit_dataset.lower()
        if not rest.lower().endswith("_" + tail):
            raise ValueError(
                f"filename '{filename}' does not end with dataset '{explicit_dataset}'. "
                f"Use --model-key explicitly or check --dataset."
            )
        model_variant = rest[: -(len(tail) + 1)]
    else:
        model_variant = "_".join(parts[1:-1])

    if not model_variant:
        raise ValueError(
            f"Could not infer model variant from '{filename}'. Use --model-key explicitly."
        )

    return f"{family}_{model_variant}", dataset


def _enum_name_from_model_key(model_key: str) -> str:
    return "_".join(part.capitalize() for part in model_key.split("_")) + "_Weights"


def _update_weights_registry(
    registry_path: Path,
    model_key: str,
    tag: str,
    repo: str,
    revision: str,
    sha: str,
    remote_path: str,
    parameter_size: int | None,
    input_size: List[int] | None,
    dataset: str,
    set_default: bool = True,
) -> Dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    model_block = registry.get(model_key, {})
    if not isinstance(model_block, dict):
        raise ValueError(f"registry entry for '{model_key}' is malformed.")

    hf_url = f"hf://{repo}@{revision}/{remote_path}"
    meta = {
        "hf_repo": repo,
        "hf_revision": revision,
        "hf_filename": remote_path,
    }
    if parameter_size is not None:
        meta["parameter_size"] = parameter_size
    if input_size is not None:
        meta["input_size"] = list(input_size)
    if dataset:
        meta["dataset"] = dataset

    entry = {
        "url": hf_url,
        "sha256": sha,
        "meta": meta.copy(),
        "dataset": dataset.lower(),
    }
    model_block[tag] = entry

    if set_default or "DEFAULT" not in model_block:
        default_entry = {
            "url": hf_url,
            "sha256": sha,
            "meta": meta.copy(),
            "dataset": dataset.lower(),
        }
        if set_default and tag != "DEFAULT":
            default_entry["default"] = "True"
        model_block.setdefault("DEFAULT", default_entry)
        model_block["DEFAULT"] = default_entry

    registry[model_key] = model_block
    registry_path.write_text(
        json.dumps(registry, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry


def _update_pyi_stub(
    pyi_path: Path, enum_name: str, tag: str, overwrite: bool = False
) -> None:
    text = pyi_path.read_text(encoding="utf-8")
    class_block_pat = re.compile(
        rf"(?ms)^class\s+{re.escape(enum_name)}\(Enum\):\n(?:    .*\n?)*"
    )
    blocks = list(class_block_pat.finditer(text))
    all_match = re.search(r"(?ms)^__all__\s*=\s*\[(.*?)\]\s*$", text)
    if all_match is None:
        raise ValueError(f"Cannot find __all__ in {pyi_path}.")

    if blocks:
        member_names: list[str] = []
        seen_members: set[str] = set()
        for block in blocks:
            body = block.group(0)
            for m in re.finditer(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*WeightEntry\s*$",
                body,
                re.MULTILINE,
            ):
                name = m.group(1)
                if name not in seen_members:
                    member_names.append(name)
                    seen_members.add(name)

        if tag and (overwrite or tag not in seen_members):
            if "DEFAULT" in seen_members:
                member_names = [m for m in member_names if m != "DEFAULT"]
                member_names.extend([tag, "DEFAULT"])
            else:
                member_names.append(tag)

        if "DEFAULT" not in seen_members:
            member_names.append("DEFAULT")
        member_names = [m for m in member_names if m != "DEFAULT"] + ["DEFAULT"]

        normalized_members: list[str] = []
        seen_members = set()
        for m in member_names:
            if m not in seen_members:
                normalized_members.append(m)
                seen_members.add(m)

        member_lines = [f"    {name}: WeightEntry" for name in normalized_members]
        new_block = f"class {enum_name}(Enum):\n" + "\n".join(member_lines) + "\n\n"

        for block in reversed(blocks):
            text = text[: block.start()] + text[block.end() :]

        all_match = re.search(r"(?ms)^__all__\s*=\s*\[(.*?)\]\s*$", text)
        if all_match is None:
            raise ValueError(f"Cannot find __all__ in {pyi_path}.")
        all_section_start = all_match.start()
        text = text[:all_section_start] + new_block + text[all_section_start:]
    else:
        # New class path.
        if f"class {enum_name}(Enum):" in text and tag and not overwrite:
            return
        insert_at = all_match.start()
        if tag:
            block_lines = [f"    {tag}: WeightEntry", "    DEFAULT: WeightEntry"]
        else:
            block_lines = ["    DEFAULT: WeightEntry"]
        new_block = (
            f"class {enum_name}(Enum):\n"
            + "\n".join(block_lines)
            + "\n\n"
        )
        text = text[:insert_at] + new_block + text[insert_at:]

    # Update __all__ while removing duplicates
    inner = all_match.group(1)
    all_items = re.findall(r'"([^"\\]+)"', inner)
    all_seen: list[str] = []
    all_set = set()
    for item in all_items:
        if item not in all_set:
            all_seen.append(item)
            all_set.add(item)
    if enum_name not in all_set:
        all_seen.append(enum_name)

    all_replacement = "__all__ = [\n" + "\n".join(
        f'    "{item}",' for item in all_seen
    ) + "\n]"
    text = re.sub(r"(?ms)^__all__\s*=\s*\[.*?\]\s*$", all_replacement, text)

    pyi_path.write_text(text, encoding="utf-8")


def _parse_input_size(v: Iterable[str] | None) -> List[int] | None:
    if v is None:
        return None
    vals = list(v)
    if len(vals) == 1 and "," in vals[0]:
        vals = [p for p in vals[0].split(",") if p != ""]
    parsed = []
    for item in vals:
        x = int(item)
        if x <= 0:
            raise ValueError("--input-size should contain positive integers.")
        parsed.append(x)
    if not parsed:
        raise ValueError("--input-size is empty.")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upload safetensors weights and sync lucid HF metadata."
    )
    p.add_argument(
        "--filename",
        help=(
            "Filename or path to .safetensors file. "
            "If a filename only is provided (no directory), it is resolved under --out-dir."
        ),
        required=True,
    )
    p.add_argument(
        "--family",
        required=True,
        help="Model family name (e.g. maskformer, resnet, ...).",
    )
    p.add_argument(
        "--repo",
        default="ChanLumerico/lucid",
        help="HF repo id or repo@revision, e.g. ChanLumerico/lucid@main",
    )
    p.add_argument(
        "--revision",
        default=os.getenv("HF_REVISION", ""),
        help="HF revision (overrides @revision in --repo if set).",
    )
    p.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN", ""),
        help="HF token for upload (or set HF_TOKEN env var).",
    )
    p.add_argument(
        "--remote-dir",
        default="weights",
        help="HF directory under repo to upload into (default: weights).",
    )
    p.add_argument(
        "--model-key",
        default="",
        help="Explicit model key (e.g. maskformer_resnet_50).",
    )
    p.add_argument(
        "--out-dir",
        default="out",
        help="Directory where weight files are located when --filename is a basename (default: out).",
    )
    p.add_argument(
        "--class",
        dest="class_name",
        default="",
        help="Explicit weights enum class name (e.g. MaskFormer_ResNet_50_Weights).",
    )
    p.add_argument(
        "--dataset",
        default="",
        help="Dataset tag (e.g. ADE20K). Auto-inferred from filename suffix.",
    )
    p.add_argument(
        "--tag",
        default="",
        help="Weight tag name for registry entry. If empty, use dataset.",
    )
    p.add_argument(
        "--parameter-size",
        type=int,
        default=0,
        help="Parameter count for meta entry.",
    )
    p.add_argument(
        "--input-size",
        nargs="+",
        help="Input size, e.g. --input-size 3 224 224",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF upload, only update local metadata files.",
    )
    p.add_argument(
        "--skip-stub",
        action="store_true",
        help="Skip updating lucid/weights/__init__.pyi stub.",
    )
    p.add_argument(
        "--commit-message",
        default="Add model weights",
        help="Commit message for Hugging Face upload.",
    )
    p.add_argument(
        "--overwrite-stub",
        action="store_true",
        help="Overwrite existing stub class definition in __init__.pyi.",
    )
    p.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repo root directory for local file updates.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes and skip write/upload operations.",
    )
    return p


def main() -> int:
    p = _build_parser()
    args = p.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = (base_dir / args.out_dir).resolve()

    if args.filename:
        candidate = Path(args.filename).expanduser()
        if (
            not candidate.is_absolute()
            and candidate.parent == Path(".")
            and (candidate.suffix.lower() == ".safetensors")
        ):
            in_file = (out_dir / candidate).resolve()
        else:
            in_file = candidate.resolve()
    else:
        if not args.model_key:
            raise ValueError(
                "When --filename is not provided, --model-key is required."
            )
        if not args.dataset:
            raise ValueError(
                "When --filename is not provided, --dataset is required."
            )
        in_file = (
            out_dir / f"{args.model_key.strip()}_{args.dataset.strip().lower()}.safetensors"
        ).resolve()

    if not in_file.exists():
        raise FileNotFoundError(f"Input weight file not found: {in_file}")
    if in_file.suffix.lower() != ".safetensors":
        raise ValueError("Only .safetensors files are supported.")

    repo, rev_from_repo = _repo_and_path(args.repo)
    revision = args.revision or rev_from_repo
    model_key, inferred_dataset = _infer_model_key(
        str(in_file.name),
        args.family,
        args.model_key.strip() or None,
        args.dataset.strip() or None,
    )

    tag = _normalize_tag(args.tag.strip() or args.dataset.strip() or inferred_dataset)
    if not tag:
        raise ValueError("Cannot infer a tag. Set --tag or --dataset explicitly.")

    remote_name = f"{args.remote_dir.rstrip('/')}/{args.family.lower()}/{in_file.name}"
    hf_url = f"hf://{repo}@{revision}/{remote_name}"
    dataset = inferred_dataset.strip().lower()

    sha = _sha256(in_file)
    parameter_size = args.parameter_size or None
    input_size = _parse_input_size(args.input_size)
    enum_name = (args.class_name.strip() or _enum_name_from_model_key(model_key))

    print(f"Target model key: {model_key}")
    print(f"Tag: {tag}")
    print(f"Remote: {remote_name}")
    print(f"SHA256: {sha}")
    print(f"HF URL: {hf_url}")
    if args.dry_run:
        print("Dry-run: no upload or file write.")
        return 0

    if not args.skip_upload:
        if not args.token and not os.getenv("HF_TOKEN"):
            print("Warning: no HF token provided; upload may fail for private repos.")
        from huggingface_hub import HfApi

        api = HfApi(token=args.token or os.getenv("HF_TOKEN"))

        api.upload_file(
            path_or_fileobj=str(in_file),
            path_in_repo=remote_name,
            repo_id=repo,
            repo_type="model",
            revision=revision,
            commit_message=args.commit_message,
        )
        print(f"Uploaded {in_file.name} to {repo}/{remote_name}")

    registry_path = base_dir / "lucid" / "weights" / "registry.json"
    pyi_path = base_dir / "lucid" / "weights" / "__init__.pyi"
    if not registry_path.exists():
        raise FileNotFoundError(f"registry.json not found: {registry_path}")
    if not pyi_path.exists():
        raise FileNotFoundError(f"__init__.pyi not found: {pyi_path}")

    if args.dry_run:
        print(f"Would update registry: {registry_path}")
        if not args.skip_stub:
            print(f"Would update stub: {pyi_path}")
        return 0

    _update_weights_registry(
        registry_path=registry_path,
        model_key=model_key,
        tag=tag,
        repo=repo,
        revision=revision,
        sha=sha,
        remote_path=remote_name,
        parameter_size=parameter_size,
        input_size=input_size,
        dataset=dataset,
        set_default=True,
    )
    print(f"Updated registry entry: {registry_path}")

    if not args.skip_stub:
        _update_pyi_stub(
            pyi_path=pyi_path,
            enum_name=enum_name,
            tag=tag,
            overwrite=args.overwrite_stub,
        )
        print(f"Updated stub: {pyi_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
