#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.numpy import load_file, save_file

import lucid.models as models
from lucid.models.text.bert import BERTConfig


@dataclass(frozen=True)
class Candidate:
    name: str
    model_key: str
    enum_name: str
    family: str
    dataset: str
    tag: str
    hf_repo: str
    lucid_ctor: str
    required_keys: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")
    min_match_ratio: float = 0.95


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="bert_mlm_base_uncased",
        model_key="bert_for_masked_lm",
        enum_name="BERTForMaskedLM_Weights",
        family="bert",
        dataset="base",
        tag="HF_BASE_UNCASED",
        hf_repo="google-bert/bert-base-uncased",
        lucid_ctor="BERTForMaskedLM",
        required_keys=("cls.predictions.decoder.bias",),
    ),
    Candidate(
        name="bert_causal_lm_base_uncased",
        model_key="bert_for_causal_lm",
        enum_name="BERTForCausalLM_Weights",
        family="bert",
        dataset="base",
        tag="HF_BASE_UNCASED",
        hf_repo="google-bert/bert-base-uncased",
        lucid_ctor="BERTForCausalLM",
        required_keys=("cls.predictions.decoder.bias",),
    ),
    Candidate(
        name="bert_nsp_base_uncased",
        model_key="bert_for_next_sentence_prediction",
        enum_name="BERTForNextSentencePrediction_Weights",
        family="bert",
        dataset="base",
        tag="HF_BASE_UNCASED",
        hf_repo="google-bert/bert-base-uncased",
        lucid_ctor="BERTForNextSentencePrediction",
        required_keys=("cls.seq_relationship.weight", "cls.seq_relationship.bias"),
    ),
    Candidate(
        name="bert_sequence_classification_sst2",
        model_key="bert_for_sequence_classification",
        enum_name="BERTForSequenceClassification_Weights",
        family="bert",
        dataset="sst2",
        tag="SST2",
        hf_repo="textattack/bert-base-uncased-SST-2",
        lucid_ctor="BERTForSequenceClassification",
        required_keys=("classifier.weight", "classifier.bias"),
    ),
    Candidate(
        name="bert_token_classification_conll03",
        model_key="bert_for_token_classification",
        enum_name="BERTForTokenClassification_Weights",
        family="bert",
        dataset="conll03",
        tag="CONLL03",
        hf_repo="dslim/bert-base-NER",
        lucid_ctor="BERTForTokenClassification",
        required_keys=("classifier.weight", "classifier.bias"),
    ),
    Candidate(
        name="bert_question_answering_squad2",
        model_key="bert_for_question_answering",
        enum_name="BERTForQuestionAnswering_Weights",
        family="bert",
        dataset="squad2",
        tag="SQUAD2",
        hf_repo="deepset/bert-base-cased-squad2",
        lucid_ctor="BERTForQuestionAnswering",
        required_keys=("qa_outputs.weight", "qa_outputs.bias"),
    ),
    Candidate(
        name="swin_tiny_imagenet1k",
        model_key="swin_tiny",
        enum_name="SwinTransformer_Tiny_Weights",
        family="swin",
        dataset="imagenet_1k",
        tag="IMAGENET1K",
        hf_repo="microsoft/swin-tiny-patch4-window7-224",
        lucid_ctor="swin_tiny",
        required_keys=(
            "head.weight",
            "norm.weight",
            "layers.0.blocks.0.attn.qkv.weight",
        ),
        min_match_ratio=0.95,
    ),
    Candidate(
        name="swin_base_imagenet1k",
        model_key="swin_base",
        enum_name="SwinTransformer_Base_Weights",
        family="swin",
        dataset="imagenet_1k",
        tag="IMAGENET1K",
        hf_repo="microsoft/swin-base-patch4-window7-224",
        lucid_ctor="swin_base",
        required_keys=(
            "head.weight",
            "norm.weight",
            "layers.0.blocks.0.attn.qkv.weight",
        ),
        min_match_ratio=0.95,
    ),
    Candidate(
        name="mask2former_swin_tiny_ade20k",
        model_key="mask2former_swin_tiny",
        enum_name="Mask2Former_Swin_Tiny_Weights",
        family="mask2former",
        dataset="ade20k",
        tag="ADE20K",
        hf_repo="facebook/mask2former-swin-tiny-ade-semantic",
        lucid_ctor="mask2former_swin_tiny",
        required_keys=(
            "class_predictor.weight",
            "model.pixel_level_module.encoder.encoder.layers.0.blocks.0.attn.qkv.weight",
        ),
        min_match_ratio=0.95,
    ),
    Candidate(
        name="mask2former_swin_small_ade20k",
        model_key="mask2former_swin_small",
        enum_name="Mask2Former_Swin_Small_Weights",
        family="mask2former",
        dataset="ade20k",
        tag="ADE20K",
        hf_repo="facebook/mask2former-swin-small-ade-semantic",
        lucid_ctor="mask2former_swin_small",
        required_keys=(
            "class_predictor.weight",
            "model.pixel_level_module.encoder.encoder.layers.0.blocks.0.attn.qkv.weight",
        ),
        min_match_ratio=0.95,
    ),
    Candidate(
        name="mask2former_swin_base_ade20k",
        model_key="mask2former_swin_base",
        enum_name="Mask2Former_Swin_Base_Weights",
        family="mask2former",
        dataset="ade20k",
        tag="ADE20K",
        hf_repo="facebook/mask2former-swin-base-ade-semantic",
        lucid_ctor="mask2former_swin_base",
        required_keys=(
            "class_predictor.weight",
            "model.pixel_level_module.encoder.encoder.layers.0.blocks.0.attn.qkv.weight",
        ),
        min_match_ratio=0.95,
    ),
    Candidate(
        name="mask2former_swin_large_ade20k",
        model_key="mask2former_swin_large",
        enum_name="Mask2Former_Swin_Large_Weights",
        family="mask2former",
        dataset="ade20k",
        tag="ADE20K",
        hf_repo="facebook/mask2former-swin-large-ade-semantic",
        lucid_ctor="mask2former_swin_large",
        required_keys=(
            "class_predictor.weight",
            "model.pixel_level_module.encoder.encoder.layers.0.blocks.0.attn.qkv.weight",
        ),
        min_match_ratio=0.95,
    ),
)


def _canon(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _collect_model_leaf_keys(node: dict[str, Any]) -> set[str]:
    out: set[str] = set()

    def _walk(cur: dict[str, Any]) -> None:
        for key, value in cur.items():
            if not isinstance(value, dict):
                continue
            if "parameter_size" in value and "submodule_count" in value:
                out.add(key)
            else:
                _walk(value)

    _walk(node)
    return out


def _select_weight_file(repo_id: str, revision: str, source_files: Iterable[str]) -> str:
    files = set(list_repo_files(repo_id, repo_type="model", revision=revision))
    for filename in source_files:
        if filename in files:
            return filename
    raise FileNotFoundError(
        f"No supported weight file found for '{repo_id}'. "
        f"Tried: {', '.join(source_files)}"
    )


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported payload type: {type(payload)}")
    return payload


def _to_numpy_state(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            out[key] = value
            continue
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy()
            continue
    return out


def _normalize_key(key: str) -> str:
    k = key
    if k.startswith("module."):
        k = k[len("module.") :]

    k = k.replace(".LayerNorm.weight", ".layernorm.weight")
    k = k.replace(".LayerNorm.bias", ".layernorm.bias")
    k = k.replace(".LayerNorm.gamma", ".layernorm.weight")
    k = k.replace(".LayerNorm.beta", ".layernorm.bias")

    if k == "cls.predictions.bias":
        k = "cls.predictions.decoder.bias"

    return k


def _map_swin_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    qkv_chunks: dict[str, dict[str, np.ndarray]] = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        if key.startswith("swin.embeddings.patch_embeddings.projection."):
            tail = key.split(".")[-1]
            out[f"patch_embed.proj.{tail}"] = value
            continue
        if key.startswith("swin.embeddings.norm."):
            tail = key.split(".")[-1]
            out[f"patch_embed.norm.{tail}"] = value
            continue
        if key.startswith("swin.layernorm."):
            tail = key.split(".")[-1]
            out[f"norm.{tail}"] = value
            continue
        if key.startswith("classifier."):
            tail = key.split(".")[-1]
            out[f"head.{tail}"] = value
            continue

        if key.startswith("swin.encoder.layers."):
            parts = key.split(".")
            if len(parts) >= 8 and parts[4] == "blocks":
                stage = parts[3]
                block = parts[5]
                sub = ".".join(parts[6:])
                base = f"layers.{stage}.blocks.{block}"

                if sub == "attention.self.relative_position_bias_table":
                    out[f"{base}.attn.relative_position_bias_table"] = value
                    continue
                if sub == "attention.self.relative_position_index":
                    out[f"{base}.attn.relative_pos_index"] = value
                    continue
                if sub in (
                    "attention.self.query.weight",
                    "attention.self.query.bias",
                    "attention.self.key.weight",
                    "attention.self.key.bias",
                    "attention.self.value.weight",
                    "attention.self.value.bias",
                ):
                    which = sub.split(".")[2]  # query|key|value
                    param = sub.split(".")[-1]  # weight|bias
                    qkv_key = f"{base}.attn.qkv.{param}"
                    qkv_chunks.setdefault(qkv_key, {})[which] = value
                    continue
                if sub.startswith("attention.output.dense."):
                    tail = sub.split(".")[-1]
                    out[f"{base}.attn.proj.{tail}"] = value
                    continue
                if sub.startswith("layernorm_before."):
                    tail = sub.split(".")[-1]
                    out[f"{base}.norm1.{tail}"] = value
                    continue
                if sub.startswith("layernorm_after."):
                    tail = sub.split(".")[-1]
                    out[f"{base}.norm2.{tail}"] = value
                    continue
                if sub.startswith("intermediate.dense."):
                    tail = sub.split(".")[-1]
                    out[f"{base}.mlp.fc1.{tail}"] = value
                    continue
                if sub.startswith("output.dense."):
                    tail = sub.split(".")[-1]
                    out[f"{base}.mlp.fc2.{tail}"] = value
                    continue

            if len(parts) >= 7 and parts[4] == "downsample":
                stage = parts[3]
                if parts[5] in ("reduction", "norm"):
                    tail = parts[-1]
                    out[f"layers.{stage}.downsample.{parts[5]}.{tail}"] = value
                    continue

    for qkv_key, chunks in qkv_chunks.items():
        if all(name in chunks for name in ("query", "key", "value")):
            out[qkv_key] = np.concatenate(
                [chunks["query"], chunks["key"], chunks["value"]], axis=0
            )

    return out


def _map_mask2former_swin_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    qkv_chunks: dict[str, dict[str, np.ndarray]] = {}

    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        if (
            ".attention.self.query." in key
            or ".attention.self.key." in key
            or ".attention.self.value." in key
        ):
            if key.endswith(".weight"):
                qkv_key = (
                    key.replace(".attention.self.query.weight", ".attn.qkv.weight")
                    .replace(".attention.self.key.weight", ".attn.qkv.weight")
                    .replace(".attention.self.value.weight", ".attn.qkv.weight")
                )
                slot = (
                    "query"
                    if ".attention.self.query." in key
                    else "key"
                    if ".attention.self.key." in key
                    else "value"
                )
                qkv_chunks.setdefault(qkv_key, {})[slot] = value
                continue

            if key.endswith(".bias"):
                qkv_key = (
                    key.replace(".attention.self.query.bias", ".attn.qkv.bias")
                    .replace(".attention.self.key.bias", ".attn.qkv.bias")
                    .replace(".attention.self.value.bias", ".attn.qkv.bias")
                )
                slot = (
                    "query"
                    if ".attention.self.query." in key
                    else "key"
                    if ".attention.self.key." in key
                    else "value"
                )
                qkv_chunks.setdefault(qkv_key, {})[slot] = value
                continue

        key = key.replace(".attention.output.dense.", ".attn.proj.")
        key = key.replace(
            ".attention.self.relative_position_bias_table",
            ".attn.relative_position_bias_table",
        )
        key = key.replace(
            ".attention.self.relative_position_index",
            ".attn.relative_pos_index",
        )
        key = key.replace(".layernorm_before.", ".norm1.")
        key = key.replace(".layernorm_after.", ".norm2.")
        key = key.replace(".intermediate.dense.", ".mlp.fc1.")
        key = key.replace(".output.dense.", ".mlp.fc2.")
        out[key] = value

    for qkv_key, chunks in qkv_chunks.items():
        if all(name in chunks for name in ("query", "key", "value")):
            out[qkv_key] = np.concatenate(
                [chunks["query"], chunks["key"], chunks["value"]], axis=0
            )

    return out


def _normalize_hf_state(
    state: dict[str, np.ndarray], candidate: Candidate
) -> dict[str, np.ndarray]:
    if candidate.family == "mask2former":
        return _map_mask2former_swin_state(state)

    if candidate.family == "swin":
        return _map_swin_state(state)

    out: dict[str, np.ndarray] = {}
    for key, value in state.items():
        out[_normalize_key(key)] = value

    if (
        "cls.predictions.decoder.weight" not in out
        and "bert.embeddings.word_embeddings.weight" in out
    ):
        out["cls.predictions.decoder.weight"] = out[
            "bert.embeddings.word_embeddings.weight"
        ]
    return out


def _build_bert_config(config: dict[str, Any]) -> BERTConfig:
    return BERTConfig(
        vocab_size=int(config["vocab_size"]),
        hidden_size=int(config["hidden_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        intermediate_size=int(config["intermediate_size"]),
        hidden_act=config["hidden_act"],
        hidden_dropout_prob=float(config["hidden_dropout_prob"]),
        attention_probs_dropout_prob=float(config["attention_probs_dropout_prob"]),
        max_position_embeddings=int(config["max_position_embeddings"]),
        tie_word_embedding=bool(config.get("tie_word_embeddings", True)),
        type_vocab_size=int(config.get("type_vocab_size", 2)),
        initializer_range=float(config["initializer_range"]),
        layer_norm_eps=float(config["layer_norm_eps"]),
        use_cache=bool(config.get("use_cache", True)),
        is_decoder=bool(config.get("is_decoder", False)),
        add_cross_attention=bool(config.get("add_cross_attention", False)),
        chunk_size_feed_forward=int(config.get("chunk_size_feed_forward", 0)),
        pad_token_id=int(config.get("pad_token_id", 0)),
        bos_token_id=config.get("bos_token_id"),
        eos_token_id=config.get("eos_token_id"),
        classifier_dropout=config.get("classifier_dropout"),
        add_pooling_layer=bool(config.get("add_pooling_layer", True)),
    )


def _extract_num_labels(config: dict[str, Any], default: int) -> int:
    num_labels = config.get("num_labels")
    if num_labels is None:
        num_labels = config.get("_num_labels")
    if num_labels is None:
        label2id = config.get("label2id")
        if isinstance(label2id, dict) and label2id:
            num_labels = len(label2id)
    if num_labels is None:
        id2label = config.get("id2label")
        if isinstance(id2label, dict) and id2label:
            num_labels = len(id2label)
    return int(num_labels if num_labels is not None else default)


def _build_mask2former_config_kwargs(hf_config: dict[str, Any]) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    fields = models.Mask2FormerConfig.__dataclass_fields__
    for key in fields:
        if key == "num_labels":
            continue
        if key in hf_config:
            model_kwargs[key] = hf_config[key]

    backbone_config = model_kwargs.get("backbone_config")
    if isinstance(backbone_config, dict):
        backbone_config = dict(backbone_config)
        if (
            backbone_config.get("model_type") == "swin"
            and int(backbone_config.get("window_size", 7)) >= 12
            and int(backbone_config.get("image_size", 224)) < 384
        ):
            # HF Swin-Base semantic checkpoints often carry image_size=224 in config
            # while using 12x12 windows. Lucid Swin expects 384 here.
            backbone_config["image_size"] = 384

        if backbone_config.get("model_type") == "swin":
            keep = {
                "model_type",
                "image_size",
                "patch_size",
                "num_channels",
                "embed_dim",
                "depths",
                "num_heads",
                "window_size",
                "drop_path_rate",
                "out_features",
                "out_indices",
                "hidden_sizes",
            }
            backbone_config = {k: v for k, v in backbone_config.items() if k in keep}

        model_kwargs["backbone_config"] = backbone_config

    if "enforce_input_projection" not in model_kwargs and "enforce_input_proj" in hf_config:
        model_kwargs["enforce_input_projection"] = hf_config["enforce_input_proj"]

    if "num_channels" not in model_kwargs:
        backbone_config = hf_config.get("backbone_config")
        if isinstance(backbone_config, dict) and "num_channels" in backbone_config:
            model_kwargs["num_channels"] = backbone_config["num_channels"]
        else:
            model_kwargs["num_channels"] = 3

    return model_kwargs


def _build_lucid_model(candidate: Candidate, hf_config: dict[str, Any]) -> Any:
    if candidate.family == "mask2former":
        ctor = getattr(models, candidate.lucid_ctor)
        num_labels = _extract_num_labels(hf_config, default=150)
        kwargs = _build_mask2former_config_kwargs(hf_config)
        return ctor(num_labels=num_labels, **kwargs)

    if candidate.family == "swin":
        ctor = getattr(models, candidate.lucid_ctor)
        num_classes = _extract_num_labels(hf_config, default=1000)
        return ctor(num_classes=num_classes)

    bert_config = _build_bert_config(hf_config)
    ctor = getattr(models, candidate.lucid_ctor)
    if candidate.lucid_ctor in (
        "BERTForSequenceClassification",
        "BERTForTokenClassification",
    ):
        num_labels = _extract_num_labels(hf_config, default=2)
        return ctor(bert_config, num_labels=num_labels)
    return ctor(bert_config)


def _download_source_state(
    candidate: Candidate, revision: str
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    config_path = hf_hub_download(candidate.hf_repo, "config.json", revision=revision)
    hf_config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    weight_name = _select_weight_file(
        candidate.hf_repo,
        revision=revision,
        source_files=candidate.source_files,
    )
    weight_path = hf_hub_download(candidate.hf_repo, weight_name, revision=revision)

    if weight_name.endswith(".safetensors"):
        state = load_file(weight_path)
    else:
        loaded = torch.load(weight_path, map_location="cpu")
        state = _to_numpy_state(_unwrap_state_dict(loaded))

    return hf_config, _normalize_hf_state(state, candidate), weight_name


def _merge_state(
    model: Any, source_state: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[str], list[str], list[str]]:
    target_state = model.state_dict()
    merged = dict(target_state)

    matched: list[str] = []
    missing: list[str] = []
    mismatched_shape: list[str] = []

    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            missing.append(key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            mismatched_shape.append(key)
            continue

        merged[key] = source_value.astype(target_value.dtype, copy=False)
        matched.append(key)

    return merged, matched, missing, mismatched_shape


def _run_upload(
    *,
    python_bin: str,
    upload_script: Path,
    in_file: Path,
    candidate: Candidate,
    repo: str,
    revision: str,
    base_dir: Path,
    parameter_size: int,
    skip_upload: bool,
    dry_run: bool,
) -> None:
    cmd = [
        python_bin,
        str(upload_script),
        "--filename",
        str(in_file),
        "--family",
        candidate.family,
        "--repo",
        repo,
        "--revision",
        revision,
        "--model-key",
        candidate.model_key,
        "--dataset",
        candidate.dataset,
        "--tag",
        candidate.tag,
        "--class",
        candidate.enum_name,
        "--parameter-size",
        str(parameter_size),
        "--base-dir",
        str(base_dir),
        "--commit-message",
        f"Import {candidate.name} weights from {candidate.hf_repo}",
    ]
    if skip_upload:
        cmd.append("--skip-upload")
    if dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"upload script failed for {candidate.name}:\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        print(proc.stdout.strip())


def _extract_model_config(model: Any) -> dict[str, Any] | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    return None


def _enrich_registry_meta(
    *,
    registry_path: Path,
    model_key: str,
    tag: str,
    family: str,
    enum_name: str,
    config: dict[str, Any] | None,
) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    model_block = registry.get(model_key)
    if not isinstance(model_block, dict):
        return

    tags = [tag, "DEFAULT"]
    for t in tags:
        entry = model_block.get(t)
        if not isinstance(entry, dict):
            continue
        meta = entry.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            entry["meta"] = meta
        meta["family"] = family
        meta["enum_name"] = enum_name
        if config is not None:
            meta["config"] = config

    registry[model_key] = model_block
    registry_path.write_text(
        json.dumps(registry, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Import HF model hub checkpoints into lucid safetensors, "
            "then sync lucid weight registry via devtools/upload_weights_to_hf.py."
        )
    )
    p.add_argument(
        "--models",
        default="auto",
        help=(
            "Comma separated candidate names to run. "
            "Use 'all' for all candidates, 'auto' for missing ones only."
        ),
    )
    p.add_argument(
        "--include-existing",
        action="store_true",
        help="Include model keys that already exist in lucid/weights/registry.json.",
    )
    p.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Stop after N successful imports (0 = no limit).",
    )
    p.add_argument(
        "--hf-revision",
        default="main",
        help="HF source revision for candidate repos.",
    )
    p.add_argument(
        "--target-repo",
        default="ChanLumerico/lucid",
        help="HF target repo where lucid weights are uploaded.",
    )
    p.add_argument(
        "--target-revision",
        default="main",
        help="HF target repo revision.",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF upload and only update local registry/stub.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery/matching only; do not save or upload.",
    )
    p.add_argument(
        "--out-dir",
        default="out",
        help="Output directory for generated .safetensors files.",
    )
    p.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Lucid repository root.",
    )
    p.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.0,
        help="Override candidate min match ratio if > 0.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    upload_script = base_dir / "devtools" / "upload_weights_to_hf.py"
    if not upload_script.exists():
        raise FileNotFoundError(f"upload script not found: {upload_script}")

    weights_registry_path = base_dir / "lucid" / "weights" / "registry.json"
    models_registry_path = base_dir / "lucid" / "models" / "registry.json"
    weights_registry = json.loads(weights_registry_path.read_text(encoding="utf-8"))
    models_registry = json.loads(models_registry_path.read_text(encoding="utf-8"))

    existing_keys = {_canon(k) for k in weights_registry.keys()}
    model_zoo_keys = {_canon(k) for k in _collect_model_leaf_keys(models_registry)}

    by_name = {c.name: c for c in CANDIDATES}
    if args.models in ("auto", "all"):
        selected = list(CANDIDATES)
    else:
        requested = [x.strip() for x in args.models.split(",") if x.strip()]
        missing = [x for x in requested if x not in by_name]
        if missing:
            raise KeyError(
                f"Unknown candidates: {missing}. "
                f"Available: {', '.join(sorted(by_name))}"
            )
        selected = [by_name[x] for x in requested]

    if args.models == "auto" and not args.include_existing:
        selected = [c for c in selected if _canon(c.model_key) not in existing_keys]

    selected = [c for c in selected if _canon(c.lucid_ctor) in model_zoo_keys]
    if not selected:
        print("No candidates selected.")
        return 0

    print("Selected candidates:")
    for c in selected:
        print(f"  - {c.name}: {c.lucid_ctor} <= {c.hf_repo}")

    successes = 0
    failures: list[tuple[str, str]] = []

    for candidate in selected:
        print(f"\n=== {candidate.name} ===")
        try:
            hf_config, source_state, source_file = _download_source_state(
                candidate, args.hf_revision
            )
            print(
                f"source repo={candidate.hf_repo} file={source_file} "
                f"keys={len(source_state)}"
            )

            model = _build_lucid_model(candidate, hf_config)
            merged, matched, missing, mismatched_shape = _merge_state(
                model, source_state
            )

            total = len(merged)
            ratio = len(matched) / total if total else 0.0
            min_ratio = (
                args.min_match_ratio
                if args.min_match_ratio > 0
                else candidate.min_match_ratio
            )
            missing_required = [k for k in candidate.required_keys if k not in matched]

            print(
                f"match: {len(matched)}/{total} ({ratio:.3f}), "
                f"missing={len(missing)}, shape_mismatch={len(mismatched_shape)}"
            )
            if missing_required:
                raise RuntimeError(
                    f"required keys missing after merge: {missing_required}"
                )
            if ratio < min_ratio:
                raise RuntimeError(
                    f"match ratio {ratio:.3f} < required {min_ratio:.3f}"
                )

            out_file = out_dir / f"{candidate.model_key}_{candidate.dataset}.safetensors"
            if args.dry_run:
                print(f"dry-run: would save {out_file}")
            else:
                save_file(merged, str(out_file))
                print(f"saved: {out_file}")

            _run_upload(
                python_bin=sys.executable,
                upload_script=upload_script,
                in_file=out_file,
                candidate=candidate,
                repo=args.target_repo,
                revision=args.target_revision,
                base_dir=base_dir,
                parameter_size=int(model.parameter_size),
                skip_upload=args.skip_upload,
                dry_run=args.dry_run,
            )

            if candidate.family == "mask2former" and not args.dry_run:
                _enrich_registry_meta(
                    registry_path=weights_registry_path,
                    model_key=candidate.model_key,
                    tag=candidate.tag,
                    family="Mask2Former",
                    enum_name=candidate.enum_name,
                    config=_extract_model_config(model),
                )
                print(f"enriched registry meta: {candidate.model_key}:{candidate.tag}")

            successes += 1
            if args.max_models > 0 and successes >= args.max_models:
                print(f"Reached --max-models={args.max_models}; stopping.")
                break

        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            print(f"FAILED: {msg}")
            failures.append((candidate.name, msg))

    print("\n=== Summary ===")
    print(f"success={successes}")
    print(f"failed={len(failures)}")
    for name, reason in failures:
        print(f"  - {name}: {reason}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
