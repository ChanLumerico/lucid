import numpy as np
import pytest

import lucid
import lucid.transforms as T


def _img(seed=0, C=3, H=8, W=8):
    return lucid.tensor(
        np.random.default_rng(seed).standard_normal((C, H, W)).astype(np.float64)
    )


def test_normalize_zero_mean_unit_std():
    x = _img(seed=1)
    mean = [float(np.asarray(x.data)[c].mean()) for c in range(3)]
    std = [float(np.asarray(x.data)[c].std()) for c in range(3)]
    out = T.Normalize(mean=mean, std=std)(x)
    arr = np.asarray(out.data)
    for c in range(3):
        np.testing.assert_allclose(arr[c].mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(arr[c].std(), 1.0, atol=1e-10)


def test_normalize_broadcast_per_channel():
    x = _img(seed=2)
    out = T.Normalize(mean=[0.5, 0.5, 0.5], std=[2.0, 2.0, 2.0])(x)
    expected = (np.asarray(x.data) - 0.5) / 2.0
    np.testing.assert_allclose(np.asarray(out.data), expected, atol=1e-12)


def test_centercrop_shape():
    x = _img(seed=3, H=10, W=10)
    out = T.CenterCrop(size=(6, 6))(x)
    assert tuple(out.shape) == (3, 6, 6)


def test_resize_smaller():
    x = _img(seed=4, H=8, W=8)
    out = T.Resize(size=(4, 4))(x)
    assert tuple(out.shape) == (3, 4, 4)
    assert np.isfinite(np.asarray(out.data)).all()


def test_random_horizontal_flip_p_zero_is_identity():
    x = _img(seed=5)
    out = T.RandomHorizontalFlip(p=0.0)(x)
    np.testing.assert_array_equal(np.asarray(out.data), np.asarray(x.data))


def test_random_horizontal_flip_p_one_is_mirror():
    x = _img(seed=6)
    out = T.RandomHorizontalFlip(p=1.0)(x)
    expected = np.asarray(x.data)[:, :, ::-1]
    np.testing.assert_array_equal(np.asarray(out.data), expected)


def test_random_vertical_flip_p_one():
    x = _img(seed=7)
    out = T.RandomVerticalFlip(p=1.0)(x)
    expected = np.asarray(x.data)[:, ::-1, :]
    np.testing.assert_array_equal(np.asarray(out.data), expected)


def test_compose_runs_in_order():
    x = _img(seed=8)
    pipe = T.Compose(
        [
            T.RandomHorizontalFlip(p=1.0),
            T.Normalize(mean=[0.0] * 3, std=[1.0] * 3),
        ]
    )
    out = pipe(x)
    expected = np.asarray(x.data)[:, :, ::-1]
    np.testing.assert_allclose(np.asarray(out.data), expected, atol=1e-12)


def test_random_crop_shape():
    x = _img(seed=9, H=10, W=10)
    out = T.RandomCrop(size=(4, 4))(x)
    assert tuple(out.shape) == (3, 4, 4)


def test_random_rotation_finite():
    x = _img(seed=10)
    out = T.RandomRotation(degrees=15)(x)
    assert tuple(out.shape) == (3, 8, 8)
    assert np.isfinite(np.asarray(out.data)).all()
