"""DenseNet weight converter — torchvision → Lucid.

Lucid's DenseNet mirrors torchvision's module layout *exactly*
(``features.conv0`` / ``features.norm0`` / ``features.denseblockN.``
``denselayerM.{norm1,conv1,norm2,conv2}`` / ``features.transitionN.`` /
``features.norm5`` / ``classifier``), so the converter is a pure
identity map — no ``map_key`` rewrites at all.

One historical torchvision quirk: pre-0.8 checkpoints used dotted
sub-keys like ``norm.1`` / ``conv.1`` inside dense layers, which a
regex used to have to normalise to ``norm1`` / ``conv1``.  The modern
``torchvision.models.densenet*(weights=...)`` builders already emit the
flattened ``norm1`` form, so no normalisation is needed here.
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_DENSENET_CITATION = (
    "@inproceedings{huang2017densely,\n"
    "  title={Densely Connected Convolutional Networks},\n"
    "  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens "
    "and Weinberger, Kilian Q.},\n"
    "  booktitle={CVPR}, year={2017}\n"
    "}"
)

_DENSENET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "densenet_121": ("densenet_121_cls", "lucid-dl/densenet-121", "DenseNet-121"),
    "densenet_161": ("densenet_161_cls", "lucid-dl/densenet-161", "DenseNet-161"),
    "densenet_169": ("densenet_169_cls", "lucid-dl/densenet-169", "DenseNet-169"),
    "densenet_201": ("densenet_201_cls", "lucid-dl/densenet-201", "DenseNet-201"),
}

_TV_BUILDERS = {
    "densenet_121": (tvm.densenet121, tvm.DenseNet121_Weights),
    "densenet_161": (tvm.densenet161, tvm.DenseNet161_Weights),
    "densenet_169": (tvm.densenet169, tvm.DenseNet169_Weights),
    "densenet_201": (tvm.densenet201, tvm.DenseNet201_Weights),
}


class DenseNetArch(Architecture):
    """Converter for one torchvision DenseNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _DENSENET_VARIANTS:
            raise KeyError(f"DenseNetArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = self._builder(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _DENSENET_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Identity — Lucid mirrors the torchvision DenseNet layout exactly.
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _DENSENET_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import ImageClassification

        tf = self._tv_weights.transforms()
        preset = ImageClassification(
            crop_size=int(tf.crop_size[0]),
            resize_size=int(tf.resize_size[0]),
            mean=tuple(float(m) for m in tf.mean),
            std=tuple(float(s) for s in tf.std),
            interpolation=str(tf.interpolation.value),
        )
        preprocessing = preset.to_dict()
        meta = {
            "num_params": int(tv_meta.get("num_params", 0)),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": dict(tv_meta.get("_metrics", {})),
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="densenet",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_DENSENET_CITATION,
            title=title,
            paper_url="Huang et al., 2017 — *Densely Connected "
            "Convolutional Networks* (arXiv:1608.06993)",
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("densenet_121")
def _121(tag: str) -> Architecture:
    return DenseNetArch("densenet_121", tag)


@register_arch("densenet_161")
def _161(tag: str) -> Architecture:
    return DenseNetArch("densenet_161", tag)


@register_arch("densenet_169")
def _169(tag: str) -> Architecture:
    return DenseNetArch("densenet_169", tag)


@register_arch("densenet_201")
def _201(tag: str) -> Architecture:
    return DenseNetArch("densenet_201", tag)
