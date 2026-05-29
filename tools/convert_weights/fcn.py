"""FCN semantic-segmentation weight converter — torchvision → Lucid.

Lucid's FCN (`lucid/models/vision/fcn/_model.py`) mirrors the torchvision
``fcn_resnet{50,101}`` layout almost exactly: a dilated-ResNet backbone
(layer3 dilation 2, layer4 dilation 4) + an ``FCNHead`` ``classifier``
(``classifier.{0,1,4}``) + an ``aux_classifier`` on the layer-3 output, and
the same forward (``classifier(c5)`` → bilinear ``interpolate`` to the input
size, ``align_corners=False``).  A full key/shape diff shows 334 keys on each
side, **zero** shape mismatches on the 328 common keys; the only divergence is
the stem naming — torchvision exposes ``backbone.conv1`` / ``backbone.bn1``
whereas Lucid wraps them in a ``stem`` Sequential (``backbone.stem.0`` /
``backbone.stem.1``).  So ``map_key`` is identity except for those 6 stem keys.

Source: ``torchvision.models.segmentation.fcn_resnet{50,101}`` with the
``COCO_WITH_VOC_LABELS_V1`` tag (21 VOC classes; COCO-val2017 images filtered
to the 20 VOC categories + background).
"""

import dataclasses
import re

import torchvision.models.segmentation as tvs

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_FCN_CITATION = (
    "@inproceedings{long2015fully,\n"
    "  title={Fully Convolutional Networks for Semantic Segmentation},\n"
    "  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},\n"
    "  booktitle={CVPR}, year={2015}\n"
    "}"
)

_FCN_VARIANTS: dict[str, tuple[str, str, str]] = {
    "fcn_resnet50": ("fcn_resnet50", "lucid-dl/fcn-resnet-50", "FCN ResNet-50"),
    "fcn_resnet101": ("fcn_resnet101", "lucid-dl/fcn-resnet-101", "FCN ResNet-101"),
}

_TV_BUILDERS = {
    "fcn_resnet50": (tvs.fcn_resnet50, tvs.FCN_ResNet50_Weights),
    "fcn_resnet101": (tvs.fcn_resnet101, tvs.FCN_ResNet101_Weights),
}


class FCNArch(Architecture):
    """Converter for one torchvision FCN variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _FCN_VARIANTS:
            raise KeyError(f"FCNArch: unknown arch {arch!r}")
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

        factory = _FCN_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Stem: torchvision backbone.conv1 / backbone.bn1 → Lucid stem Sequential.
        if src_key == "backbone.conv1.weight":
            return "backbone.stem.0.weight"
        m = re.match(r"backbone\.bn1\.(.+)", src_key)
        if m:
            return f"backbone.stem.1.{m.group(1)}"
        # Everything else is identity (backbone.layer*, classifier.*, aux_classifier.*).
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _FCN_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import Segmentation

        tf = self._tv_weights.transforms()
        resize = int(getattr(tf, "resize_size", [520])[0]) if isinstance(
            getattr(tf, "resize_size", 520), (list, tuple)
        ) else int(getattr(tf, "resize_size", 520) or 520)
        preset = Segmentation(
            crop_size=resize,
            resize_size=resize,
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
            task="semantic-segmentation",
            model_type="fcn",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_FCN_CITATION,
            title=title,
            paper_url="Long et al., 2015 — *Fully Convolutional Networks "
            "for Semantic Segmentation* (arXiv:1411.4038)",
            categories=categories,
            datasets=["coco", "pascal-voc"],
            meta=meta,
        )


@register_arch("fcn_resnet50")
def _fcn50(tag: str) -> Architecture:
    return FCNArch("fcn_resnet50", tag)


@register_arch("fcn_resnet101")
def _fcn101(tag: str) -> Architecture:
    return FCNArch("fcn_resnet101", tag)
