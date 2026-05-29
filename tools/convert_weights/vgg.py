"""VGG weight converter — torchvision → Lucid.

Lucid's VGG ``features`` trunk mirrors torchvision's ``features``
``Sequential`` index-for-index (Conv → [BN] → ReLU → MaxPool), so every
``features.*`` key — including the BatchNorm ``running_mean`` /
``running_var`` / ``num_batches_tracked`` buffers in the ``*_bn``
variants — is a pure identity map.

Only the fully-connected head is laid out differently.  torchvision
wraps the head in a single ``classifier`` ``Sequential``::

    classifier.0 = Linear(25088, 4096)   # FC6
    classifier.3 = Linear(4096, 4096)    # FC7
    classifier.6 = Linear(4096, num_cls) # final

while Lucid names the three Linear layers explicitly:

    ==============================  =====================
    torchvision                     Lucid
    ==============================  =====================
    ``classifier.0.{weight,bias}``  ``fc6.{weight,bias}``
    ``classifier.3.{weight,bias}``  ``fc7.{weight,bias}``
    ``classifier.6.{weight,bias}``  ``classifier.{weight,bias}``
    ==============================  =====================

All eight checkpoints (VGG-11/13/16/19, with / without BatchNorm) use
the standard ImageNet eval pipeline (224 crop / 256 resize / bilinear /
ImageNet stats).
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_VGG_CITATION = (
    "@inproceedings{simonyan2015very,\n"
    "  title={Very Deep Convolutional Networks for Large-Scale Image "
    "Recognition},\n"
    "  author={Simonyan, Karen and Zisserman, Andrew},\n"
    "  booktitle={ICLR}, year={2015}\n"
    "}"
)

_VGG_PAPER_URL = (
    "Simonyan & Zisserman, 2015 — *Very Deep Convolutional Networks for "
    "Large-Scale Image Recognition* (arXiv:1409.1556)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_VGG_VARIANTS: dict[str, tuple[str, str, str]] = {
    "vgg_11": ("vgg_11_cls", "lucid-dl/vgg-11", "VGG-11"),
    "vgg_13": ("vgg_13_cls", "lucid-dl/vgg-13", "VGG-13"),
    "vgg_16": ("vgg_16_cls", "lucid-dl/vgg-16", "VGG-16"),
    "vgg_19": ("vgg_19_cls", "lucid-dl/vgg-19", "VGG-19"),
    "vgg_11_bn": ("vgg_11_bn_cls", "lucid-dl/vgg-11-bn", "VGG-11-BN"),
    "vgg_13_bn": ("vgg_13_bn_cls", "lucid-dl/vgg-13-bn", "VGG-13-BN"),
    "vgg_16_bn": ("vgg_16_bn_cls", "lucid-dl/vgg-16-bn", "VGG-16-BN"),
    "vgg_19_bn": ("vgg_19_bn_cls", "lucid-dl/vgg-19-bn", "VGG-19-BN"),
}

# torchvision builders keyed by arch.
_TV_BUILDERS = {
    "vgg_11": (tvm.vgg11, tvm.VGG11_Weights),
    "vgg_13": (tvm.vgg13, tvm.VGG13_Weights),
    "vgg_16": (tvm.vgg16, tvm.VGG16_Weights),
    "vgg_19": (tvm.vgg19, tvm.VGG19_Weights),
    "vgg_11_bn": (tvm.vgg11_bn, tvm.VGG11_BN_Weights),
    "vgg_13_bn": (tvm.vgg13_bn, tvm.VGG13_BN_Weights),
    "vgg_16_bn": (tvm.vgg16_bn, tvm.VGG16_BN_Weights),
    "vgg_19_bn": (tvm.vgg19_bn, tvm.VGG19_BN_Weights),
}


class VGGArch(Architecture):
    """Converter for one torchvision VGG variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VGG_VARIANTS:
            raise KeyError(f"VGGArch: unknown arch {arch!r}")
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

        factory = _VGG_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # FC head: torchvision packs the three Linear layers into one
        # ``classifier`` Sequential; Lucid names them fc6 / fc7 / classifier.
        if src_key.startswith("classifier.0."):
            return "fc6." + src_key[len("classifier.0.") :]
        if src_key.startswith("classifier.3."):
            return "fc7." + src_key[len("classifier.3.") :]
        if src_key.startswith("classifier.6."):
            return "classifier." + src_key[len("classifier.6.") :]
        # Everything in the conv trunk (incl. BatchNorm buffers) is identity.
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _VGG_VARIANTS[self.arch]
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
            model_type="vgg",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_VGG_CITATION,
            title=title,
            paper_url=_VGG_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("vgg_11")
def _build_vgg_11(tag: str) -> Architecture:
    return VGGArch("vgg_11", tag)


@register_arch("vgg_13")
def _build_vgg_13(tag: str) -> Architecture:
    return VGGArch("vgg_13", tag)


@register_arch("vgg_16")
def _build_vgg_16(tag: str) -> Architecture:
    return VGGArch("vgg_16", tag)


@register_arch("vgg_19")
def _build_vgg_19(tag: str) -> Architecture:
    return VGGArch("vgg_19", tag)


@register_arch("vgg_11_bn")
def _build_vgg_11_bn(tag: str) -> Architecture:
    return VGGArch("vgg_11_bn", tag)


@register_arch("vgg_13_bn")
def _build_vgg_13_bn(tag: str) -> Architecture:
    return VGGArch("vgg_13_bn", tag)


@register_arch("vgg_16_bn")
def _build_vgg_16_bn(tag: str) -> Architecture:
    return VGGArch("vgg_16_bn", tag)


@register_arch("vgg_19_bn")
def _build_vgg_19_bn(tag: str) -> Architecture:
    return VGGArch("vgg_19_bn", tag)
