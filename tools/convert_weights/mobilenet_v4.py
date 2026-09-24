"""MobileNet v4 weight converter — timm → Lucid.

Lucid's MobileNet v4 mirrors timm's ``mobilenetv4_*`` module layout
*exactly*, so the converter is a pure identity map:

==============================================  ============================
timm                                            Lucid
==============================================  ============================
``conv_stem`` / ``bn1``                          (identical)
``blocks.S.B.conv`` / ``.bn1``    (Conv2D rows)  (identical)
``blocks.S.B.{conv_exp,bn1,conv_pwl,bn2}``       (identical, fused IB)
``blocks.S.B.{dw_start,pw_exp,dw_mid,pw_proj}``  (identical, UIB;
``.{conv,bn}``                                    ``dw_*`` absent when off)
``blocks.S.B.layer_scale.gamma``  (hybrids)      (identical, ``(C,)``)
``blocks.S.B.norm`` +                            (identical, Mobile MQA)
``blocks.S.B.attn.{query,key,value,output}.*``
``conv_head`` / ``norm_head`` / ``classifier``   (identical)
==============================================  ============================

Source checkpoints are timm's ImageNet-1k weights for the five paper
variants (Qin et al., "MobileNetV4: Universal Models for the Mobile
Ecosystem", ECCV 2024).  The tag is the timm pretrained tag, upper-cased
(``E2400_R224_IN1K`` → ``mobilenetv4_conv_small.e2400_r224_in1k``).  Those
recipes are timm's, not the paper's, so accuracy metadata is left for the
publisher to measure rather than copied from the paper's Table 6.
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_MOBILENET_V4_CITATION = (
    "@inproceedings{qin2024mobilenetv4,\n"
    "  title={MobileNetV4: Universal Models for the Mobile Ecosystem},\n"
    "  author={Qin, Danfeng and Leichner, Chas and Delakis, Manolis and "
    "Fornoni, Marco and Luo, Shixin and Yang, Fan and Wang, Weijun and "
    "Banbury, Colby and Ye, Chengxi and Akin, Berkin and others},\n"
    "  booktitle={ECCV}, year={2024}\n"
    "}"
)

_MOBILENET_V4_PAPER_URL = (
    "Qin et al., 2024 — *MobileNetV4: Universal Models for the Mobile "
    "Ecosystem* (arXiv:2404.10518)"
)

# arch -> (lucid_cls_factory, timm_arch, repo_id, title)
_MOBILENET_V4_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "mobilenet_v4_conv_small": (
        "mobilenet_v4_conv_small_cls",
        "mobilenetv4_conv_small",
        "lucid-dl/mobilenet-v4-conv-small",
        "MobileNet-v4-Conv-Small",
    ),
    "mobilenet_v4_conv_medium": (
        "mobilenet_v4_conv_medium_cls",
        "mobilenetv4_conv_medium",
        "lucid-dl/mobilenet-v4-conv-medium",
        "MobileNet-v4-Conv-Medium",
    ),
    "mobilenet_v4_conv_large": (
        "mobilenet_v4_conv_large_cls",
        "mobilenetv4_conv_large",
        "lucid-dl/mobilenet-v4-conv-large",
        "MobileNet-v4-Conv-Large",
    ),
    "mobilenet_v4_hybrid_medium": (
        "mobilenet_v4_hybrid_medium_cls",
        "mobilenetv4_hybrid_medium",
        "lucid-dl/mobilenet-v4-hybrid-medium",
        "MobileNet-v4-Hybrid-Medium",
    ),
    "mobilenet_v4_hybrid_large": (
        "mobilenet_v4_hybrid_large_cls",
        "mobilenetv4_hybrid_large",
        "lucid-dl/mobilenet-v4-hybrid-large",
        "MobileNet-v4-Hybrid-Large",
    ),
}


class MobileNetV4Arch(Architecture):
    """Converter for one timm MobileNet v4 variant + tag.

    timm's key layout matches Lucid's exactly, so :meth:`map_key` is a
    pure identity — the safety gates in :func:`convert` still verify the
    1:1 key set, per-key shapes, and a real ``load_state_dict``.
    """

    def __init__(self, arch: str, tag: str) -> None:
        import timm

        if arch not in _MOBILENET_V4_VARIANTS:
            raise KeyError(f"MobileNetV4Arch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory_name, timm_arch, _, _ = _MOBILENET_V4_VARIANTS[arch]
        self._timm_name = f"{timm_arch}.{tag.lower()}"
        self._model = timm.create_model(self._timm_name, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_model = models.create_model(factory_name)

    def source_state_dict(self) -> dict[str, object]:
        return {
            k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()
        }

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Identity — Lucid mirrors the timm MobileNet v4 layout exactly.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, _, repo_id, title = _MOBILENET_V4_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        cfg = self._model.default_cfg
        from lucid.utils.transforms import ImageClassification

        crop = int(cfg["input_size"][1])
        resize = int(round(crop / float(cfg.get("crop_pct", 0.875))))
        preset = ImageClassification(
            crop_size=crop,
            resize_size=resize,
            mean=tuple(float(m) for m in cfg.get("mean", (0.485, 0.456, 0.406))),
            std=tuple(float(s) for s in cfg.get("std", (0.229, 0.224, 0.225))),
            interpolation=str(cfg.get("interpolation", "bicubic")),
        )

        meta = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": str(cfg.get("url", "")),
            # Measure before publishing: these are timm's recipes, and the
            # paper's Table 6 numbers describe different training runs.
            "metrics": {"ImageNet-1k": {}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="mobilenet_v4",
            source=f"timm/{self._timm_name}",
            license=str(cfg.get("license", "apache-2.0")),
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preset.to_dict(),
            citation=_MOBILENET_V4_CITATION,
            title=title,
            paper_url=_MOBILENET_V4_PAPER_URL,
            categories=[],
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("mobilenet_v4_conv_small")
def _build_mobilenet_v4_conv_small(tag: str) -> Architecture:
    return MobileNetV4Arch("mobilenet_v4_conv_small", tag)


@register_arch("mobilenet_v4_conv_medium")
def _build_mobilenet_v4_conv_medium(tag: str) -> Architecture:
    return MobileNetV4Arch("mobilenet_v4_conv_medium", tag)


@register_arch("mobilenet_v4_conv_large")
def _build_mobilenet_v4_conv_large(tag: str) -> Architecture:
    return MobileNetV4Arch("mobilenet_v4_conv_large", tag)


@register_arch("mobilenet_v4_hybrid_medium")
def _build_mobilenet_v4_hybrid_medium(tag: str) -> Architecture:
    return MobileNetV4Arch("mobilenet_v4_hybrid_medium", tag)


@register_arch("mobilenet_v4_hybrid_large")
def _build_mobilenet_v4_hybrid_large(tag: str) -> Architecture:
    return MobileNetV4Arch("mobilenet_v4_hybrid_large", tag)
