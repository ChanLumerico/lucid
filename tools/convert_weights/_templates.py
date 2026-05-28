"""Fixed-schema renderers for ``config.json`` and the Hub model card.

Both artifacts are generated from a :class:`~tools.convert_weights._base.
ConversionSpec` so every converted checkpoint carries identical,
predictable structure.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.convert_weights._base import ConversionSpec


def render_config_json(spec: "ConversionSpec") -> str:
    """Render the per-tag ``config.json`` (HF-compatible, self-contained).

    Includes the Lucid architecture config, the inference preprocessing
    spec, a ``weights`` provenance/metrics block, and ``id2label`` /
    ``label2id`` maps (the HF vision-classification convention).
    """
    id2label = {str(i): name for i, name in enumerate(spec.categories)}
    label2id = {name: i for i, name in enumerate(spec.categories)}

    payload: dict[str, object] = {
        "library_name": "lucid",
        "architecture": spec.architecture,
        "model_type": spec.model_type,
        "task": spec.task,
        "config": spec.config,
        "preprocessing": spec.preprocessing,
        "weights": {
            "tag": spec.tag,
            "num_classes": spec.num_classes,
            "license": spec.license,
            "source": spec.source,
            **spec.meta,
        },
    }
    if spec.categories:
        payload["id2label"] = id2label
        payload["label2id"] = label2id

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _metrics_rows(spec: "ConversionSpec") -> str:
    """Build the markdown tag-comparison table row(s) from ``meta``."""
    metrics = spec.meta.get("metrics", {})
    acc1 = acc5 = "—"
    if isinstance(metrics, dict):
        for _dataset, vals in metrics.items():
            if isinstance(vals, dict):
                acc1 = str(vals.get("acc@1", "—"))
                acc5 = str(vals.get("acc@5", "—"))
            break
    params = spec.meta.get("num_params", "—")
    params_m = f"{params / 1e6:.1f}M" if isinstance(params, (int, float)) else "—"
    gflops = spec.meta.get("gflops", "—")
    size = spec.meta.get("file_size_mb", "—")
    size_s = f"{size} MB" if size != "—" else "—"
    src = spec.source.split("/")[0]
    return (
        f"| `{spec.tag}` *(default)* | {acc1} | {acc5} | {params_m} | "
        f"{gflops} | {size_s} | {src} |"
    )


def _model_index(spec: "ConversionSpec") -> str:
    """Build the YAML ``model-index`` block from metrics."""
    metrics = spec.meta.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return ""
    dataset, vals = next(iter(metrics.items()))
    if not isinstance(vals, dict):
        return ""
    metric_lines = "\n".join(
        f"          - {{ type: {k}, value: {v} }}" for k, v in vals.items()
    )
    ds_type = str(dataset).lower().replace(" ", "-")
    return (
        "model-index:\n"
        f"  - name: {spec.architecture.replace('_', '-')}\n"
        "    results:\n"
        f"      - task: {{ type: {spec.task} }}\n"
        f"        dataset: {{ name: {dataset}, type: {ds_type} }}\n"
        "        metrics:\n"
        f"{metric_lines}\n"
    )


def render_model_card(spec: "ConversionSpec") -> str:
    """Render the repo-level ``README.md`` model card.

    YAML frontmatter (``library_name``, license, tags, datasets,
    ``pipeline_tag``, ``model-index``) + a tag comparison table + usage
    snippet + conversion provenance + license + citation.
    """
    metrics = spec.meta.get("metrics", {})
    # spec.datasets wins (lets a converter list every dataset the checkpoint
    # touched — e.g. ``["imagenet-22k", "imagenet-1k"]`` for an in22k → in1k
    # finetune); empty falls back to the metrics' first key for the simple
    # single-dataset case.
    dataset_names: list[str] = list(spec.datasets)
    if not dataset_names and isinstance(metrics, dict) and metrics:
        dataset_names = [str(next(iter(metrics))).lower().replace(" ", "-")]

    enum_name = "".join(p.capitalize() for p in spec.architecture.split("_")) + "Weights"
    paper_line = (
        f"\n> {spec.paper_url}\n" if spec.paper_url else ""
    )

    frontmatter = [
        "---",
        "library_name: lucid",
        f"license: {spec.license}",
        "tags:",
        f"  - {spec.task}",
        f"  - {spec.architecture.split('_')[0]}",
        "  - lucid",
    ]
    if dataset_names:
        frontmatter.append("datasets:")
        frontmatter += [f"  - {d}" for d in dataset_names]
    frontmatter += [f"pipeline_tag: {spec.task}"]
    mi = _model_index(spec)
    if mi:
        frontmatter.append(mi.rstrip("\n"))
    frontmatter.append("---")

    body = f"""
# {spec.title}
{paper_line}
[Lucid](https://github.com/ChanLumerico/lucid) port of `{spec.source}`,
converted to Lucid-native safetensors.

## Available weights

| Tag | acc@1 | acc@5 | Params | GFLOPs | Size | Source |
|---|---|---|---|---|---|---|
{_metrics_rows(spec)}

## Usage

```python
import lucid.models as models
from lucid.models.vision.resnet import {enum_name}

# default tag
model = models.{spec.model_name}(pretrained=True)

# explicit tag (enum or string)
model = models.{spec.model_name}(weights={enum_name}.{spec.tag})
model = models.{spec.model_name}(pretrained="{spec.tag}")

# preprocessing travels with the weights
weights = {enum_name}.{spec.tag}
preprocess = weights.transforms()
logits = model(preprocess(image)[None]).logits
```

## Conversion

Converted from `{spec.source}` via
`python -m tools.convert_weights {spec.architecture} --tag {spec.tag}`.
Key mapping + numerical parity verified against the source.

## License

`{spec.license}` — inherited from the original weights.

## Citation

```
{spec.citation}
```
"""
    return "\n".join(frontmatter) + "\n" + body
