"""RandomErasing — Zhong et al., 2017 (arXiv:1708.04896).

Unit contract:

* shape preservation across single / batched inputs
* ``p=0`` is identity, ``p=1`` always erases
* erased pixels match ``value`` exactly (constant fill)
* ``value="random"`` injects new values (output != input over erase region)
* ``value=(...)`` per-channel fill respects channel ordering
* invalid arguments raise ``ValueError`` early
* ``manual_seed`` produces reproducible erase rectangles
"""

import pytest

import lucid
import lucid.utils.transforms as T


class TestRandomErasing:
    def test_shape_preserved_single(self) -> None:
        tf = T.RandomErasing(p=1.0, scale=(0.1, 0.2))
        out = tf(lucid.rand(3, 64, 64))
        assert tuple(out.shape) == (3, 64, 64)

    def test_shape_preserved_batch(self) -> None:
        tf = T.RandomErasing(p=1.0, scale=(0.1, 0.2))
        out = tf(lucid.rand(2, 3, 64, 64))
        assert tuple(out.shape) == (2, 3, 64, 64)

    def test_p_zero_is_identity(self) -> None:
        tf = T.RandomErasing(p=0.0)
        x = lucid.rand(3, 32, 32)
        out = tf(x)
        assert float((out - x).abs().max().item()) == 0.0

    def test_p_one_changes_image(self) -> None:
        # Use a large erase region so it definitely changes the image.
        lucid.manual_seed(0)
        tf = T.RandomErasing(p=1.0, scale=(0.2, 0.3), value=0.0)
        x = lucid.ones(3, 64, 64)
        out = tf(x)
        # At least some pixels became 0.0 (the erase region).
        assert float(out.min().item()) == 0.0
        # And some pixels stayed at 1.0 (outside the erase region).
        assert float(out.max().item()) == 1.0

    def test_constant_fill_value(self) -> None:
        lucid.manual_seed(0)
        tf = T.RandomErasing(p=1.0, scale=(0.15, 0.25), value=0.5)
        x = lucid.ones(3, 32, 32)
        out = tf(x)
        # Erased pixels are 0.5, others are 1.0.
        unique = sorted(set(out.numpy().reshape(-1).tolist()))
        assert unique == [0.5, 1.0]

    def test_per_channel_tuple_fill(self) -> None:
        lucid.manual_seed(0)
        tf = T.RandomErasing(p=1.0, scale=(0.15, 0.25), value=(0.1, 0.2, 0.3))
        x = lucid.ones(3, 32, 32)
        out = tf(x).numpy()
        # The erased region in each channel should equal the channel's value.
        # Sample the centre column of the erased rectangle by finding a pixel
        # that's not 1.0 in channel 0.
        chan0 = out[0]
        chan1 = out[1]
        chan2 = out[2]
        erase_mask = chan0 != 1.0  # bool ndarray
        # Where channel 0 was erased, channels 1 and 2 must be erased too,
        # each with their respective fill value (float32 tolerance).
        assert float(chan0[erase_mask].min()) == pytest.approx(0.1, abs=1e-6)
        assert float(chan0[erase_mask].max()) == pytest.approx(0.1, abs=1e-6)
        assert float(chan1[erase_mask].min()) == pytest.approx(0.2, abs=1e-6)
        assert float(chan2[erase_mask].min()) == pytest.approx(0.3, abs=1e-6)

    def test_random_fill(self) -> None:
        lucid.manual_seed(0)
        tf = T.RandomErasing(p=1.0, scale=(0.15, 0.25), value="random")
        x = lucid.zeros(3, 32, 32)
        out = tf(x)
        # Random fill injects nonzero values into the erase region.
        assert float(out.abs().max().item()) > 0.0

    def test_reproducible_with_seed(self) -> None:
        tf = T.RandomErasing(p=1.0, scale=(0.1, 0.2), value=0.5)
        x = lucid.ones(3, 32, 32)
        lucid.manual_seed(42)
        out1 = tf(x).numpy()
        lucid.manual_seed(42)
        out2 = tf(x).numpy()
        # Identical seeds → identical erase rectangles.
        assert (out1 == out2).all()

    def test_invalid_scale(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            T.RandomErasing(scale=(0.5, 0.2))
        with pytest.raises(ValueError, match="scale"):
            T.RandomErasing(scale=(-0.1, 0.3))
        with pytest.raises(ValueError, match="scale"):
            T.RandomErasing(scale=(0.1, 1.5))

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="ratio"):
            T.RandomErasing(ratio=(0.0, 1.0))
        with pytest.raises(ValueError, match="ratio"):
            T.RandomErasing(ratio=(2.0, 1.0))

    def test_invalid_value_string(self) -> None:
        with pytest.raises(ValueError, match="value string"):
            T.RandomErasing(value="mean")  # type: ignore[arg-type]

    def test_value_tuple_wrong_length(self) -> None:
        tf = T.RandomErasing(p=1.0, scale=(0.15, 0.25), value=(0.1, 0.2))
        with pytest.raises(ValueError, match="tuple length"):
            tf(lucid.ones(3, 32, 32))

    def test_repr(self) -> None:
        tf = T.RandomErasing(p=0.25, scale=(0.02, 0.33), value=0.5)
        r = repr(tf)
        assert "RandomErasing" in r
        assert "p=0.25" in r
        assert "scale=(0.02, 0.33)" in r
