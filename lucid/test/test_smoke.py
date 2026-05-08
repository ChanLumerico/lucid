"""Bottom-of-the-pyramid smoke tests — run before any other tier.

These tests are deliberately tiny: their job is to fail fast and loud
when the test infrastructure itself is broken (missing fixtures,
broken conftest, busted import path).  When this module is red,
everything downstream is meaningless.
"""

import pytest

import lucid


@pytest.mark.smoke
def test_lucid_imports() -> None:
    assert hasattr(lucid, "Tensor")
    assert hasattr(lucid, "tensor")
    assert hasattr(lucid, "manual_seed")


@pytest.mark.smoke
def test_consecutive_rand_calls_diverge() -> None:
    """The autouse seed hook runs before this body, so two consecutive
    ``rand()`` calls inside one test produce a deterministic stream
    that nevertheless advances — two draws shouldn't be identical."""
    a = lucid.rand(4)
    b = lucid.rand(4)
    assert not (a.numpy() == b.numpy()).all()


@pytest.mark.smoke
def test_seed_resets_between_tests_part1() -> None:
    sample = lucid.rand(8).numpy()
    test_seed_resets_between_tests_part1.last_sample = sample.tolist()  # type: ignore[attr-defined]


@pytest.mark.smoke
def test_seed_resets_between_tests_part2() -> None:
    """Re-running the same first-call after the autouse reset must
    yield the same sample as part1 — proves determinism is in force."""
    sample = lucid.rand(8).numpy()
    expected = test_seed_resets_between_tests_part1.last_sample  # type: ignore[attr-defined]
    assert sample.tolist() == expected, (
        "RNG state did not reset between tests — autouse fixture broken"
    )


@pytest.mark.smoke
def test_device_fixture(device: str) -> None:
    """Sanity: the ``device`` fixture yields a recognised device name."""
    assert device in {"cpu", "metal"}


@pytest.mark.smoke
def test_float_dtype_fixture(float_dtype: lucid.dtype) -> None:
    assert float_dtype in (lucid.float32, lucid.float64)


@pytest.mark.smoke
def test_tensor_factory_fixture(tensor_factory) -> None:  # type: ignore[no-untyped-def]
    t = tensor_factory((3, 4))
    assert t.shape == (3, 4)
