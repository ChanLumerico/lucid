import argparse, json, hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


p = argparse.ArgumentParser()

p.add_argument("model_key")
p.add_argument("path")
p.add_argument("--repo", default="ChanLumerico/lucid")
p.add_argument("--filename", default=None)
p.add_argument("--revision", default="main")
p.add_argument("--tag", default="DEFAULT")
p.add_argument(
    "--default",
    action="store_true",
    help="Mark this tag as the DEFAULT weight for the given model key",
)
p.add_argument("--dataset", default=None)
p.add_argument("--input_size", default=None)
p.add_argument("--registry", default="../lucid/weights/registry.json")

args = p.parse_args()

model_key = args.model_key
safetensors = Path(args.path)
fn = args.filename or safetensors.name
h = sha256(safetensors)

reg_path = Path(args.registry)
reg_path.parent.mkdir(parents=True, exist_ok=True)
reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}

meta = {"hf_repo": args.repo, "hf_revision": args.revision, "hf_filename": fn}
if args.input_size:
    meta["input_size"] = [int(x) for x in args.input_size.split(",")]

entry = {
    "url": f"hf://{args.repo}@{args.revision}/weights/{model_key}/{fn}",
    "sha256": h,
    "meta": meta,
}

if args.dataset:
    entry["dataset"] = args.dataset

reg.setdefault(args.model_key, {})[args.tag] = entry
if args.default:
    entry["default"] = "True"
    reg[args.model_key]["DEFAULT"] = {k: v for k, v in entry.items() if k != "default"}

reg_path.write_text(json.dumps(reg, indent=2) + "")

print(json.dumps({args.model_key: {args.tag: entry}}, indent=2))
