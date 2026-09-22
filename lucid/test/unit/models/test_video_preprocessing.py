"""What the video preset does, pinned where it is exact.

The V-JEPA families ship a preprocessing preset rather than the no-op
they carried at first: without one a caller holding a decoded clip has
no way to reach the numbers the published weights expect.

Three parts of the pipeline were compared against the released
``VJEPA2VideoProcessor`` and agree **exactly**, and those are what this
file pins:

* the resize target — ``int(crop * 256 / 224)``, which truncates.
  ``round`` gives 293 for a 256 crop where the release gives 292, and
  one pixel there moves the crop window and rescales every pixel in it;
* the centre-crop window — read back through a ramp, both take rows and
  columns 18..273 out of 292;
* the normalisation — 2e-7 against the released mean and std.

What is *not* pinned is the resized pixel values. Lucid's bilinear
resize and the reference's differ by a sub-pixel sampling phase: on a
smooth frame the gap is ~2.6e-2 in normalised units, on white noise it
is ~2.7, which is the signature of a phase offset rather than a wrong
convention — noise decorrelates under any shift and smooth content does
not. The same magnitudes appear when an *image* preset is compared the
same way, so it is a property of the resize, not of this preset.
Asserting a bound on it here would pin Lucid's interpolation, which is
a separate question from whether this preset is assembled correctly.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
from lucid.models.generative.vjepa2_ac import VJEPA2ACWeights
from lucid.models.vision.vjepa2 import (
    VJEPA2ViTGiant384Weights,
    VJEPA2ViTLargeWeights,
)


class TestReleasedGeometry:
    """Numbers taken from the released processor's own config."""

    @pytest.mark.parametrize(("crop", "resize"), [(256, 292), (384, 438)])
    def test_the_resize_target_truncates(self, crop: int, resize: int) -> None:
        """``int(256 * 256 / 224)`` is 292; rounding 292.57 gives 293."""
        assert T.VideoClassification(crop_size=crop).resize_size == resize

    def test_the_crop_takes_the_centre_window(self) -> None:
        """A ramp encodes each pixel's (row, col) so the window reads back."""
        side = 292
        rows = np.arange(side)[None, :, None] * 1000
        cols = np.arange(side)[None, None, :]
        frame = (rows + cols).astype(np.float32)
        clip = np.ascontiguousarray(
            np.broadcast_to(frame, (3, side, side))[None].repeat(2, axis=0)
        )

        cropped = T.CenterCrop(256, 256)(T.Image(lucid.from_numpy(clip))).data
        top_left = int(cropped[0, 0, 0, 0].item())
        bottom_right = int(cropped[0, 0, -1, -1].item())
        assert (top_left // 1000, top_left % 1000) == (18, 18)
        assert (bottom_right // 1000, bottom_right % 1000) == (273, 273)

    def test_the_statistics_are_imagenet_s(self) -> None:
        preset = T.VideoClassification(crop_size=256)
        assert preset.mean == (0.485, 0.456, 0.406)
        assert preset.std == (0.229, 0.224, 0.225)

    def test_a_constant_frame_normalises_by_hand(self) -> None:
        """Isolates the arithmetic from anything spatial."""
        preset = T.VideoClassification(crop_size=64)
        clip = lucid.ones(2, 3, 100, 100) * (128.0 / 255.0)
        out = preset(T.Image(clip)).data.numpy()
        wanted = (128 / 255 - np.array(preset.mean)) / np.array(preset.std)
        for channel in range(3):
            assert np.abs(out[:, channel] - wanted[channel]).max() < 1e-5


class TestClipContract:
    """A clip in, a clip out, and nothing done across time."""

    def test_the_frame_count_survives(self) -> None:
        preset = T.VideoClassification(crop_size=64)
        out = preset(T.Image(lucid.rand(7, 3, 90, 120))).data
        assert tuple(out.shape) == (7, 3, 64, 64)

    def test_each_frame_is_handled_alone(self) -> None:
        """No temporal mixing: one frame's pixels cannot reach another."""
        preset = T.VideoClassification(crop_size=64)
        clip = lucid.rand(3, 3, 90, 120)
        together = preset(T.Image(clip)).data.numpy()
        alone = preset(T.Image(clip[1:2])).data.numpy()
        assert np.abs(together[1] - alone[0]).max() == 0.0

    def test_a_non_square_frame_keeps_its_aspect(self) -> None:
        """Shortest side to the target, then the square crop."""
        preset = T.VideoClassification(crop_size=256)
        out = preset(T.Image(lucid.rand(2, 3, 180, 320))).data
        assert tuple(out.shape) == (2, 3, 256, 256)

    def test_the_preset_round_trips_through_its_config(self) -> None:
        preset = T.VideoClassification(crop_size=384)
        back = T.AutoTransformsPreset.from_dict(preset.to_dict())
        assert isinstance(back, T.VideoClassification)
        assert back.crop_size == 384 and back.resize_size == 438


class TestWeightsCarryIt:
    """A published tag has to hand the caller its preprocessing."""

    @pytest.mark.parametrize(
        ("entry", "crop"),
        [
            (VJEPA2ViTLargeWeights.DEFAULT, 256),
            (VJEPA2ViTGiant384Weights.DEFAULT, 384),
            (VJEPA2ACWeights.DEFAULT, 256),
        ],
    )
    def test_the_tag_ships_the_released_preset(self, entry: object, crop: int) -> None:
        preset = getattr(entry, "entry").transforms
        assert isinstance(preset, T.VideoClassification), (
            "the tag ships no video preprocessing — a caller with a decoded "
            "clip cannot reach the numbers these weights expect"
        )
        assert preset.crop_size == crop

    def test_the_preset_agrees_with_the_recorded_metadata(self) -> None:
        """``meta`` and the preset must not describe different pipelines."""
        for entry, in (
            (VJEPA2ViTLargeWeights.DEFAULT,),
            (VJEPA2ViTGiant384Weights.DEFAULT,),
            (VJEPA2ACWeights.DEFAULT,),
        ):
            record = getattr(entry, "entry")
            preprocessing = record.meta["preprocessing"]
            assert isinstance(preprocessing, dict)
            assert record.transforms.crop_size == preprocessing["size"]
            assert list(record.transforms.mean) == preprocessing["mean"]
            assert list(record.transforms.std) == preprocessing["std"]
