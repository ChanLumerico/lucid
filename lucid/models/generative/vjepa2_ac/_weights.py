"""Pretrained SafeTensors declaration for V-JEPA 2-AC."""

from lucid.utils.transforms import VideoClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

__all__ = ["VJEPA2ACWeights"]

_PRESET = VideoClassification(crop_size=256)
_REQUIRES_CONFIG: dict[str, object] = {
    "image_size": 256,
    "patch_size": 16,
    "tubelet_size": 2,
    "in_channels": 3,
    "encoder_dim": 1408,
    "encoder_depth": 40,
    "encoder_heads": 22,
    "mlp_ratio": 48.0 / 11.0,
    "predictor_mlp_ratio": 4.0,
    "predictor_dim": 1024,
    "predictor_depth": 24,
    "predictor_heads": 16,
    "action_dim": 7,
    "state_dim": 7,
    "extrinsics_dim": 6,
}


@register_weights("vjepa2_ac_vit_giant")
@register_weights("vjepa2_ac_vit_giant_world_model")
class VJEPA2ACWeights(WeightsEnum):
    r"""Official V-JEPA 2-AC ViT-g/16 action-conditioned weights.

    Converted from the released ``vjepa2-ac-vitg.pt``, whose ``encoder``
    and ``predictor`` mappings load strictly into this family with nothing
    missing on either side.  Checked by running the official repository's
    own code against the same checkpoint: ``8.5e-5`` on the encoder,
    ``4.5e-5`` on the predictor, relative.  The file is the post-trained
    ViT-g encoder beside the 24-block action predictor, 1,317,394,944
    parameters together.
    """

    OFFICIAL = WeightEntry(
        url=f"{HUB_BASE}/vjepa2-ac-vitg/resolve/main/OFFICIAL/model.safetensors",
        sha256="67b4fac83e0d1f37b4777e1f3e70afe3520e5a59dcdccf0f22fe5b3838d410ce",
        num_classes=1408,
        transforms=_PRESET,
        requires_config=_REQUIRES_CONFIG,
        meta={
            "tag": "OFFICIAL",
            "source": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt",
            "license": "mit",
            "num_params": 1_317_394_944,
            "file_size_mb": 5025.54,
            "encoder_num_heads": 22,
            "predictor_num_heads": 16,
            "preprocessing": {
                "type": "video-with-actions",
                "frames_per_clip": 64,
                "size": 256,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "action_dim": 7,
                "state_dim": 7,
            },
        },
    )
    DEFAULT = OFFICIAL
