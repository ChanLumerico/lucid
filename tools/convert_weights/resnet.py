"""ResNet weight converter — torchvision → Lucid.

Maps torchvision's ResNet ``state_dict`` keys onto Lucid's module
layout.  Only the stem and head differ; the four stages
(``layerN.*``) share identical naming, so the mapping is small:

==============================  =========================
torchvision                     Lucid
==============================  =========================
``conv1.weight``                ``stem.0.weight``
``bn1.{w,b,rm,rv,nbt}``         ``stem.1.{w,b,rm,rv,nbt}``
``layerN.*``                    ``layerN.*``  (identical)
``fc.{weight,bias}``            ``classifier.{weight,bias}``
==============================  =========================
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_RESNET_CITATION = (
    "@inproceedings{he2016deep,\n"
    "  title={Deep Residual Learning for Image Recognition},\n"
    "  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},\n"
    "  booktitle={CVPR}, year={2016}\n"
    "}"
)

# Lucid factory + torchvision (builder, weights-enum) per architecture.
_RESNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "resnet_18": ("resnet_18_cls", "lucid-dl/resnet-18", "ResNet-18"),
}

# torchvision builders keyed by arch.
_TV_BUILDERS = {
    "resnet_18": (tvm.resnet18, tvm.ResNet18_Weights),
}


class ResNetArch(Architecture):
    """Converter for one ResNet variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _RESNET_VARIANTS:
            raise KeyError(f"ResNetArch: unknown arch {arch!r}")
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

        factory = _RESNET_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        if src_key == "conv1.weight":
            return "stem.0.weight"
        if src_key.startswith("bn1."):
            return "stem.1." + src_key[len("bn1.") :]
        if src_key.startswith("fc."):
            return "classifier." + src_key[len("fc.") :]
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _RESNET_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        tf = self._tv_weights.transforms()
        preprocessing = {
            "type": "ImageClassification",
            "resize_size": int(tf.resize_size[0]),
            "crop_size": int(tf.crop_size[0]),
            "mean": [float(m) for m in tf.mean],
            "std": [float(s) for s in tf.std],
            "interpolation": str(tf.interpolation.value),
        }
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
            model_type="resnet",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_RESNET_CITATION,
            title=title,
            paper_url="He et al., 2015 — *Deep Residual Learning for Image "
            "Recognition* (arXiv:1512.03385)",
            categories=categories,
            meta=meta,
        )


@register_arch("resnet_18")
def _build_resnet_18(tag: str) -> Architecture:
    return ResNetArch("resnet_18", tag)
