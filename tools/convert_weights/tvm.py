"""TVM weight converter — the authors' training pickles → Lucid.

The three released checkpoints (``lumaai/tvm`` on the Hub) are not state
dicts.  Each is a pickle of the training loop's evaluation EMA, written
with the ``torch_utils.persistence`` scheme StyleGAN introduced: every
decorated class is stored *with the source of its module*, and unpickling
``exec``\\s that source.  Loading one the ordinary way therefore needs the
authors' repository importable — and its ``training/dit.py`` imports a
Triton attention kernel at module level, which does not build on Apple
Silicon.

Nothing here runs that source.  :class:`_ParameterUnpickler` resolves the
persistence hook, and every class outside the reference framework's own
``nn`` and storage machinery, to an inert record of its pickled state, so
the file loads into a tree of plain attribute dictionaries whose leaves
are the real tensors.  :func:`_flatten` walks ``_parameters`` /
``_buffers`` / ``_modules`` through it into the usual dotted names.  As a
side effect the conversion executes no code from the checkpoint beyond
the framework's tensor rebuilding.

Two checks guard the parts a key map cannot see:

* **The training flags match the factory.**  The wrapper pickles its
  ``do_scale_param`` and the network its ``learn_guidance``; a checkpoint
  whose scaled parameterisation disagrees with the config it is loaded
  into would produce velocities off by a factor of ``w`` with no shape
  error to say so.
* **The positional table matches Lucid's.**  The reference stores its
  frozen sine-cosine table as a parameter; Lucid rebuilds it from the
  config.  The stored one is compared against
  :func:`~lucid.models.generative._common._transformers.sincos_position_embedding`
  and then dropped, so a convention drift (the column-first ordering)
  fails here rather than as a quietly degraded sample.

Licence: the Hub card says Apache-2.0 but the ``LICENSE`` file beside the
weights, and the code repository's, is CC-BY-NC-SA-4.0.  The converted
card carries the stricter of the two.
"""

import dataclasses
import pickle
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lucid.models.generative._common._transformers import sincos_position_embedding
from lucid.models.generative.tvm import TVMConfig, TVMModel
from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    "@inproceedings{zhou2026terminal,\n"
    "  title={Terminal Velocity Matching},\n"
    "  author={Zhou, Linqi and Parger, Mathias and Haque, Ayaan and "
    "Song, Jiaming},\n"
    "  booktitle={International Conference on Learning Representations "
    "(ICLR)},\n"
    "  year={2026},\n"
    "  eprint={2511.19797},\n"
    "  archivePrefix={arXiv}\n"
    "}"
)

_REPO = "lumaai/tvm"


@dataclasses.dataclass(frozen=True)
class _Release:
    """One released checkpoint and the Lucid configuration it loads into."""

    factory: str
    filename: str
    overrides: dict[str, object]
    guidance: float
    fid: dict[str, float]


# Table 6 and the authors' ``configs/*.yaml``.  The ``w = 2.25`` 512-pixel
# run in the paper was never released.
_RELEASES: dict[str, _Release] = {
    "IMAGENET1K_256_W2": _Release(
        factory="tvm_imagenet256",
        filename="im256_w2.pkl",
        overrides={},
        guidance=2.0,
        fid={"fid_1nfe": 3.29, "fid_2nfe": 2.80},
    ),
    "IMAGENET1K_256_W1P75": _Release(
        factory="tvm_imagenet256",
        filename="im256_w1p75.pkl",
        overrides={"guidance_scale": 1.75, "time_embedding_init": "normal"},
        guidance=1.75,
        fid={"fid_1nfe": 4.58, "fid_2nfe": 2.61, "fid_4nfe": 1.99},
    ),
    "IMAGENET1K_512_W2P5": _Release(
        factory="tvm_imagenet512",
        filename="im512_w2p5.pkl",
        overrides={},
        guidance=2.5,
        fid={"fid_1nfe": 4.32, "fid_2nfe": 3.50},
    ),
}

# The base configurations of the two factories, mirrored from
# ``lucid/models/generative/tvm/_pretrained.py``.
_FACTORY_CONFIGS: dict[str, dict[str, object]] = {
    "tvm_imagenet256": {},
    "tvm_imagenet512": {
        "sample_size": 64,
        "guidance_scale": 2.5,
        "scaled_parameterization": False,
        "time_embedding_init": "normal",
        "latent_mean": (1.56, -0.695, 0.483, 0.729),
        "latent_std": (5.27, 5.91, 4.21, 4.31),
    },
}


# ── loading a persistence pickle without its source ──────────────────────────


class _Record(dict[str, Any]):
    """Inert stand-in for any class the converter refuses to import.

    A ``dict`` so that dictionary subclasses (``EasyDict``) unpickle their
    items into it; a ``__setstate__`` so that objects unpickle their
    attributes into ``__dict__``.  Either way nothing of the original
    class runs.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple):
            state = state[0]
        if isinstance(state, dict):
            self.__dict__.update(state)


def _reconstruct_persistent(meta: dict[str, Any]) -> _Record:
    """Replacement for ``torch_utils.persistence._reconstruct_persistent_obj``.

    The original ``exec``\\s ``meta["module_src"]`` and instantiates the
    class it defines; this keeps only ``meta["state"]``, the object's
    ``__dict__`` — which, for a network, is its parameters and children.
    """
    record = _Record()
    record.__setstate__(meta.get("state") or {})
    return record


# Modules whose classes are safe to resolve for real: the tensor and
# storage rebuilders and the framework's own layers, whose instances carry
# the ``_parameters`` / ``_modules`` dictionaries this walk reads.
_REAL_PREFIXES = (
    "torch._utils",
    "torch._tensor",
    "torch.storage",
    "torch.nn.",
    "torch.serialization",
)
_REAL_NAMES = {
    ("torch", "Size"),
    ("torch", "device"),
    ("torch", "float32"),
    ("torch", "float16"),
    ("torch", "bfloat16"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("torch", "BFloat16Storage"),
    ("torch", "LongStorage"),
    ("torch", "IntStorage"),
    ("torch", "BoolStorage"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("copyreg", "_reconstructor"),
    ("copyreg", "__newobj__"),
    ("copyreg", "__newobj_ex__"),
    ("builtins", "object"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "slice"),
}


class _ParameterUnpickler(pickle.Unpickler):
    """Resolve only what a parameter tree needs; stub everything else."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("torch_utils.persistence", "_reconstruct_persistent_obj"):
            return _reconstruct_persistent
        if (module, name) in _REAL_NAMES or module.startswith(_REAL_PREFIXES):
            return super().find_class(module, name)
        if module == "numpy" or module.startswith("numpy."):
            return super().find_class(module, name)
        return _Record


def _attrs(obj: object) -> dict[str, Any]:
    """An unpickled object's attribute dictionary."""
    return cast(dict[str, Any], getattr(obj, "__dict__", {}))


def _flatten(obj: object, prefix: str, out: dict[str, np.ndarray]) -> None:
    """Collect every parameter and buffer under ``obj`` by dotted name."""
    attrs = _attrs(obj)
    for group in ("_parameters", "_buffers"):
        for name, tensor in (attrs.get(group) or {}).items():
            if tensor is not None:
                out[prefix + name] = tensor.detach().to(torch.float32).cpu().numpy()
    for name, child in (attrs.get("_modules") or {}).items():
        if child is not None:
            _flatten(child, f"{prefix}{name}.", out)


def _load_ema(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """The evaluation EMA's tensors, and the wrapper's and network's flags."""
    with open(path, "rb") as handle:
        data = _ParameterUnpickler(handle).load()
    if not isinstance(data, dict) or "ema" not in data:
        raise RuntimeError(f"convert(tvm): {path} holds no 'ema' entry")
    wrapper = data["ema"]
    state: dict[str, np.ndarray] = {}
    _flatten(wrapper, "", state)
    network = (_attrs(wrapper).get("_modules") or {}).get("model")
    return state, _attrs(wrapper), _attrs(network)


# ── the recipe ───────────────────────────────────────────────────────────────


class TVMArch(Architecture):
    """Conversion recipe for one released TVM checkpoint."""

    def __init__(self, model_name: str, tag: str) -> None:
        """Bind a factory name to one of its releases.

        Parameters
        ----------
        model_name : str
            Lucid factory the weights load into.
        tag : str
            A key of ``_RELEASES`` whose factory is ``model_name``.

        Raises
        ------
        ValueError
            If ``tag`` names no release of ``model_name``.
        """
        mine = sorted(k for k, r in _RELEASES.items() if r.factory == model_name)
        if tag not in mine:
            raise ValueError(f"no {model_name} release tagged {tag!r}; known: {mine}")
        self.model_name = model_name
        self.tag = tag
        self.release = _RELEASES[tag]

    def _config(self) -> TVMConfig:
        fields = {**_FACTORY_CONFIGS[self.model_name], **self.release.overrides}
        return TVMConfig(**cast(dict[str, Any], fields))

    def source_state_dict(self) -> dict[str, object]:
        """Download the pickle, load its EMA, and check what keys cannot.

        Returns
        -------
        dict
            Upstream names to arrays, ``pos_embed`` already verified.

        Raises
        ------
        RuntimeError
            If the checkpoint's scaled parameterisation or guidance
            embedder disagrees with the target config, or its positional
            table differs from the one Lucid builds.
        """
        path = hf_hub_download(_REPO, self.release.filename)
        state, wrapper, network = _load_ema(path)
        config = self._config()

        scaled = bool(wrapper.get("do_scale_param", False))
        if scaled != config.scaled_parameterization:
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} was trained with "
                f"do_scale_param={scaled}, but the target config has "
                f"scaled_parameterization={config.scaled_parameterization}"
            )
        if not network.get("learn_guidance", True):
            raise RuntimeError(
                f"convert(tvm): {self.release.filename} has no guidance "
                f"embedder; Lucid's TVM always conditions on 1/w"
            )

        stored = state.pop("model.pos_embed", None)
        if stored is not None:
            sample = config.sample_size
            side = (sample if isinstance(sample, int) else sample[0]) // config.patch_size
            built = sincos_position_embedding(config.hidden_size, side).numpy()
            if not np.allclose(stored, built, atol=1e-5):
                raise RuntimeError(
                    "convert(tvm): the checkpoint's positional table differs "
                    "from Lucid's sincos_position_embedding — the axis order "
                    "or frequency ladder has drifted"
                )
        return cast(dict[str, object], state)

    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        return TVMModel(self._config())

    def map_key(self, src_key: str) -> str | None:
        """Map one upstream key to its Lucid name.

        Parameters
        ----------
        src_key : str
            Upstream name, ``model.``-prefixed as the wrapper stores it.

        Returns
        -------
        str or None
            The Lucid name, or ``None`` for anything outside the network.
        """
        if not src_key.startswith("model."):
            return None
        key = src_key[len("model.") :]
        leaf = key.rsplit(".", 1)[-1]
        simple = {
            "x_embedder.proj.": "patch_embed.",
            "t_embedder.mlp.0.": "time_embed.fc1.",
            "t_embedder.mlp.2.": "time_embed.fc2.",
            "s_embedder.mlp.0.": "interval_embed.fc1.",
            "s_embedder.mlp.2.": "interval_embed.fc2.",
            "guidance_embedder.mlp.0.": "guidance_embed.fc1.",
            "guidance_embedder.mlp.2.": "guidance_embed.fc2.",
            "y_embedder.embedding_table.": "label_embed.",
            "final_layer.linear.": "final.proj.",
            "final_layer.adaLN_modulation.1.": "final.ada_ln.1.",
        }
        for upstream, lucid_prefix in simple.items():
            if key == upstream + leaf:
                return lucid_prefix + leaf
        if key.startswith("blocks."):
            index, rest = key.split(".", 2)[1], key.split(".", 2)[2]
            inner = {
                "attn.qkv.": "attn.qkv.",
                "attn.q_norm.": "attn.q_norm.",
                "attn.k_norm.": "attn.k_norm.",
                "attn.proj.": "attn.proj.",
                "mlp.fc1.": "mlp.fc1.",
                "mlp.fc2.": "mlp.fc2.",
                "adaLN_modulation.1.": "ada_ln.1.",
            }
            for upstream, lucid_prefix in inner.items():
                if rest == upstream + leaf:
                    return f"blocks.{index}.{lucid_prefix}{leaf}"
        return None

    def spec(self) -> ConversionSpec:
        """Return the static metadata written beside the weights."""
        config = self._config()
        pixels = 256 if config.sample_size == 32 else 512
        return ConversionSpec(
            model_name=self.model_name,
            architecture="tvm",
            repo_id="lucid-dl/tvm",
            tag=self.tag,
            task="image-generation",
            model_type="tvm",
            source=f"https://huggingface.co/{_REPO}/blob/main/{self.release.filename}",
            license="cc-by-nc-sa-4.0",
            num_classes=config.num_classes,
            config=dataclasses.asdict(config),
            preprocessing={
                "resize": pixels,
                "rescale": 1 / 127.5,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            citation=_CITATION,
            title=f"TVM-XL/2 ({pixels}x{pixels}, w={self.release.guidance})",
            paper_url="https://arxiv.org/abs/2511.19797",
            categories=["generative", "flow-map", "class-conditional"],
            datasets=["imagenet-1k"],
            meta={
                "num_params": 678_068_496,
                "metrics": {"imagenet-1k": dict(self.release.fid)},
                "guidance_scale": self.release.guidance,
                "latent_size": config.sample_size,
                "vae": "stabilityai/sd-vae-ft-ema",
                "latent_whitening": {
                    "mean": list(config.latent_mean),
                    "std": list(config.latent_std),
                },
                "sampler": "x_s = x_t + (s - t) * model(x_t, t, s), t on linspace(1, 0, n+1)",
            },
        )


@register_arch("tvm_imagenet256")
def _tvm_256(tag: str) -> TVMArch:
    """Build the recipe for a 256-pixel release."""
    return TVMArch("tvm_imagenet256", tag)


@register_arch("tvm_imagenet512")
def _tvm_512(tag: str) -> TVMArch:
    """Build the recipe for a 512-pixel release."""
    return TVMArch("tvm_imagenet512", tag)
