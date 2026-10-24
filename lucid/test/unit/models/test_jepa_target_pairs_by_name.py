"""The JEPA families' target average pairs each parameter with its namesake.

``update_target`` zips the context encoder's ``parameters()`` with the
target encoder's by position.  ``Module.__setattr__`` used to move a name
it re-assigned to the end of its registry, and ``qkv.weight += 0.0`` under
``no_grad`` is such a re-assignment: the two encoders' orders came apart
and the average mixed ``qkv.weight`` into ``qkv.bias`` — a shape error on
a tiny I-JEPA, a silently wrong average wherever two neighbours' shapes
agree (LCD-551).  Checked here name by name, so a reorder anywhere shows
up as the wrong value under the right name rather than as a shape error.
"""

import pytest

import lucid
from lucid.models import create_model

# Each family's encoder-backed factory, overridden down to a tiny config.
_FAMILIES = {
    "ijepa": (
        "ijepa_base_16",
        dict(
            image_size=32,
            patch_size=8,
            dim=16,
            depth=1,
            num_heads=2,
            predictor_dim=8,
            predictor_depth=1,
            min_keep=1,
            target_scale=(0.1, 0.2),
            context_scale=(0.8, 1.0),
        ),
    ),
    "vjepa": (
        "vjepa_large_16",
        dict(
            image_size=32,
            patch_size=8,
            tubelet_size=2,
            num_frames=4,
            dim=24,
            depth=1,
            num_heads=2,
            predictor_dim=12,
            predictor_depth=1,
            short_range_blocks=2,
            long_range_blocks=1,
        ),
    ),
    "vjepa2": (
        "vjepa2_vit_large",
        dict(
            image_size=16,
            patch_size=8,
            tubelet_size=2,
            num_frames=4,
            dim=24,
            depth=1,
            num_heads=2,
            predictor_dim=12,
            predictor_depth=1,
            predictor_heads=2,
            predictor_num_mask_tokens=2,
        ),
    ),
}


def _snapshot(module: lucid.nn.Module) -> dict[str, lucid.Tensor]:
    return {name: p.detach().clone() for name, p in module.named_parameters()}


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_the_target_average_pairs_parameters_by_name(family: str) -> None:
    factory, overrides = _FAMILIES[family]
    model = create_model(factory, **overrides)
    qkv = model.encoder.get_submodule("blocks.0.attn.qkv")
    with lucid.no_grad():
        qkv.weight += 0.0  # an assignment, as far as the module can tell
        for parameter in model.encoder.parameters():
            parameter[:] = parameter + 1.0
    live = _snapshot(model.encoder)
    start = _snapshot(model.target_encoder)

    model.update_target(0.25)

    for name, averaged in model.target_encoder.named_parameters():
        expected = 0.25 * start[name] + 0.75 * live[name]
        assert float((averaged - expected).abs().max().item()) < 1e-5, name
