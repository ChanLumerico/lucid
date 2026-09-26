"""A compiled call that runs eagerly says so — once per kind of reason.

``lucid.compile`` falls back to eager whenever it cannot build an
executable, and the answer is right either way. That made the fallback
easy to miss entirely: a bfloat16 model, which the MPSGraph emitters have
no type for, ran eagerly on every call and said nothing unless
``LUCID_COMPILE_VERBOSE=1`` was set. Each fallback now goes through one
notice that warns the first time a kind of reason is met in a process and
stays quiet after, since a model that falls back does so on every call.

bfloat16 is also caught before the trace rather than after it: tracing
the whole forward only for the lowering to refuse the first operation was
work spent to learn what the dtype already said.
"""

import warnings
from collections.abc import Iterator

import pytest

import lucid
import lucid.nn as nn
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


@pytest.fixture(autouse=True)
def _fresh_notices(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts as a process that has announced nothing yet."""
    monkeypatch.setattr(fallback, "_NOTICED", set())
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    yield


def _fallback_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, fallback.CompileFallbackWarning)
    ]


class TestTheNotice:
    def test_a_category_warns_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
            fallback._fallback_notice("lowering", "op 'qr' has no emitter")
        said = _fallback_warnings(caught)
        assert len(said) == 1
        assert "op 'inv' has no emitter" in said[0]
        assert "running eagerly" in said[0]

    def test_each_category_warns_on_its_own(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
            fallback._fallback_notice("bfloat16", "bfloat16 is not lowered")
        assert len(_fallback_warnings(caught)) == 2

    def test_verbose_prints_every_one_and_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LUCID_COMPILE_VERBOSE", "1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "first")
            fallback._fallback_notice("lowering", "second")
        assert not _fallback_warnings(caught)
        err = capsys.readouterr().err
        assert "[compile] eager fallback: first" in err
        assert "[compile] eager fallback: second" in err

    def test_the_warning_points_at_the_caller(self) -> None:
        """Not at a line inside ``lucid.compile``, which the reader cannot act on."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback._fallback_notice("lowering", "op 'inv' has no emitter")
        (only,) = [
            w for w in caught if issubclass(w.category, fallback.CompileFallbackWarning)
        ]
        assert only.filename == __file__


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
class TestBfloat16:
    def _model(self) -> nn.Module:
        lucid.manual_seed(0)
        return nn.Linear(4, 3).to(COMPILE_DEVICE).to(lucid.bfloat16).eval()

    def test_it_runs_eagerly_and_says_so_once(self) -> None:
        model = self._model()
        compiled = lucid.compile(model)
        other = lucid.compile(self._model())
        x = lucid.randn(2, 4).to(COMPILE_DEVICE).to(lucid.bfloat16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = compiled(x)
            compiled(x)
            other(x)
        said = _fallback_warnings(caught)
        assert len(said) == 1, said
        assert "bfloat16" in said[0]
        assert first.dtype == lucid.bfloat16
        expected = model(x)
        assert bool((first == expected).all().item())
        assert compiled.cache_info()["eager_only"]

    def test_nothing_is_traced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Refused before the trace, not by the lowering after it."""

        def _no_trace() -> None:
            raise AssertionError("a bfloat16 call was traced")

        monkeypatch.setattr(lucid.compile, "_tracing", _no_trace)
        compiled = lucid.compile(self._model())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = compiled(lucid.randn(2, 4).to(COMPILE_DEVICE).to(lucid.bfloat16))
        assert out.shape == (2, 3)

    def test_a_bfloat16_argument_alone_is_enough(self) -> None:
        compiled = lucid.compile(nn.Identity().to(COMPILE_DEVICE).eval())
        x = lucid.randn(2, 4).to(COMPILE_DEVICE).to(lucid.bfloat16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compiled(x)
        said = _fallback_warnings(caught)
        assert len(said) == 1 and "bfloat16" in said[0]


@pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")
def test_a_write_to_an_argument_is_announced() -> None:
    class _Bumps(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            with lucid.no_grad():
                x.add_(1.0)
            return self.lin(x)

    compiled = lucid.compile(_Bumps().to(COMPILE_DEVICE))
    x = lucid.zeros(2, 4, device=COMPILE_DEVICE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled(x)
        compiled(x)
    said = _fallback_warnings(caught)
    assert len(said) == 1
    assert "own arguments" in said[0]
