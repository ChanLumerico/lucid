"""``eval()`` twice on one input has to give one answer.

The zoo had no test for this. It is the only axis that sees a
regularisation layer which never learns it is at inference: a dropout
whose probability is forwarded unconditionally keeps dropping in
``eval()``, and nothing else in the suite notices. Shapes agree, the
training step still converges, published-weight parity still passes —
that comparison feeds tensors to sub-modules and reads the numbers once,
so a *stochastic* forward and a deterministic one are indistinguishable
to it unless you call twice.

V-JEPA 2 shipped with exactly that defect: ``dropout_p=attn_drop_rate``
straight into ``scaled_dot_product_attention``, which applies it with
``training=True`` regardless of the module's mode. It was invisible at
the default ``attn_drop_rate=0.0`` and would have surfaced as
"the same clip scores differently every call".

The families are the ones ``test_models_train_step.py`` already builds
at unit-test size, reused rather than re-declared: that table is the
zoo's list of "buildable through a public factory with a known small
config", and keeping one copy means a family added there is covered
here too.

Dropout is turned *on* where the config exposes it. With the default
0.0 the branch under test never runs, so a test at defaults would pass
against the broken code — which is why the original defect survived
every gate the zoo had.
"""

import dataclasses
from typing import Any

import pytest

import lucid
import lucid.models as models
from lucid.test.unit.models.test_models_train_step import CASES, _family_of


def _values_of(case: Any) -> tuple[Any, ...]:
    """The case tuple, whether or not pytest.param wrapped it in marks."""
    return tuple(case.values) if hasattr(case, "values") else tuple(case)


def _tensors_of(output: Any) -> list[tuple[str, lucid.Tensor]]:
    """Every tensor a forward answered with, named for the message.

    The zoo's outputs are ``@dataclass(slots=True)``, which have no
    ``__dict__`` — ``vars()`` raises on some and returns nothing useful
    on others.  Detectors and a few backbones answer with a bare tensor
    or a tuple instead.
    """
    if isinstance(output, lucid.Tensor):
        return [("output", output)]
    if isinstance(output, (tuple, list)):
        return [
            (f"[{i}]", v) for i, v in enumerate(output) if isinstance(v, lucid.Tensor)
        ]
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        found = []
        for field in dataclasses.fields(output):
            value = getattr(output, field.name, None)
            if isinstance(value, lucid.Tensor):
                found.append((field.name, value))
            elif isinstance(value, (tuple, list)):
                found += [
                    (f"{field.name}[{i}]", v)
                    for i, v in enumerate(value)
                    if isinstance(v, lucid.Tensor)
                ]
        return found
    return []


#: Config fields that put a dropout somewhere in the forward.  A family
#: that declares none is still worth running: the point is that *no*
#: source of non-determinism survives ``eval()``, and dropout is only
#: the one we have already been bitten by.
_DROPOUT_FIELDS = (
    "attn_drop_rate",
    "attention_dropout",
    "attention_probs_dropout_prob",
    "drop_rate",
    "dropout",
    "hidden_dropout_prob",
    "drop_path_rate",
)

#: Families whose forward is legitimately stochastic even at inference —
#: they draw, and the draw is the point.  Naming one here is a claim
#: that its randomness is the model's, not a leaked training-mode layer.
_SAMPLES_BY_DESIGN: dict[str, str] = {
    "vae": "reparameterised sampling is the forward",
    "hvae": "reparameterised sampling is the forward",
}


def _with_dropout_on(factory: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Raise every dropout the factory's config declares off its default."""
    config = models.AutoConfig.from_pretrained(factory)
    enabled = dict(overrides)
    for field in _DROPOUT_FIELDS:
        if not hasattr(config, field):
            continue
        current = getattr(config, field)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            enabled[field] = 0.5
    return enabled


@pytest.mark.parametrize(
    ("family", "factory", "overrides", "make_inputs"),
    CASES,
    ids=[_family_of(c) for c in CASES],
)
def test_eval_is_deterministic(
    family: str,
    factory: str,
    overrides: dict[str, Any],
    make_inputs: Any,
) -> None:
    """Two calls, one input, one answer — with dropout turned up."""
    if family in _SAMPLES_BY_DESIGN:
        pytest.skip(f"{family}: {_SAMPLES_BY_DESIGN[family]}")

    lucid.manual_seed(0)
    model = models.create_model(factory, **_with_dropout_on(factory, overrides))
    model.eval()
    args, kwargs = make_inputs()

    with lucid.no_grad():
        first = model(*args, **kwargs)
        second = model(*args, **kwargs)

    left, right = _tensors_of(first), _tensors_of(second)
    assert left, f"{factory}: no tensor in {type(first).__name__} to compare"
    assert [n for n, _ in left] == [n for n, _ in right]
    for (name, a), (_, b) in zip(left, right):
        worst = float((a - b).abs().max().item())
        assert worst == 0.0, (
            f"{factory}.{name} differs between two eval() calls by {worst:g} — "
            "something in the forward is still drawing. A dropout handed its "
            "probability to a functional call without gating on self.training "
            "is the usual cause."
        )


def test_the_dropout_really_was_turned_on() -> None:
    """Guards the test above.

    At the zoo's default of 0.0 every dropout branch is skipped, so the
    determinism assertion would hold against code that never gates on
    training mode at all — which is how the defect this file exists for
    survived.  At least one case must actually raise a dropout.
    """
    raised = []
    for case in CASES:
        _family, factory, overrides, _make = _values_of(case)
        if _with_dropout_on(factory, overrides) != overrides:
            raised.append(factory)
    assert raised, "no case in CASES exposes a dropout field; this file proves nothing"
