"""Every distribution accepts exactly the reference's parameters and support.

A distribution whose constraint is stricter than the reference's refuses a
parameter the reference accepts — ``Poisson(rate=0)`` was a ``ValueError``
under the default ``validate_args`` — and one whose constraint is looser
constructs happily and then answers NaN.  Either way the closed forms at the
edge of the domain become unreachable or wrong without anything raising.

The comparison is by behaviour, not by constraint class: each candidate is
offered to both libraries and the two verdicts must agree.

* ``test_parameter_domain`` builds the distribution with the candidate —
  Lucid under its default validation, the reference with
  ``validate_args=True`` — so the domain compared includes checks that live
  outside a constraint table (the reference's ``Geometric`` refuses
  ``probs == 0`` in its constructor, ``Uniform`` refuses ``low >= high``).
* ``test_support`` offers candidate values to ``support.check``.
* ``test_constraint_table`` compares the names in ``arg_constraints`` and the
  ``is_discrete`` / ``event_dim`` of the support.

``_EXCEPTIONS`` lists every difference kept on purpose, with the reason.  A
listed difference must still be there — fixing one fails until its entry is
removed — and a difference that is not listed fails outright.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.distributions as D

pytestmark = [
    pytest.mark.parity,
    # The reference's Wishart warns about a small df on the way to refusing it.
    pytest.mark.filterwarnings("ignore:Low df values detected:UserWarning"),
]

_INF = math.inf
_NAN = math.nan

# Scalar candidates: both sides of 0 and 1, a non-integer above 1, an integer
# above 1, the infinities and NaN.
_SCALARS: list[float] = [-_INF, -1.0, -0.5, 0.0, 0.3, 1.0, 1.5, 3.0, _INF, _NAN]

_PROB_VECTORS: list[list[float]] = [
    [0.2, 0.3, 0.5],
    [0.0, 0.5, 0.5],
    [1.0, 0.0, 0.0],
    [0.5, 0.6, 0.1],
    [2.0, 3.0, 5.0],
    [-0.1, 0.6, 0.5],
    [0.0, 0.0, 0.0],
    [_INF, 1.0, 1.0],
    [_NAN, 0.5, 0.5],
]

_LOGIT_VECTORS: list[list[float]] = [
    [0.0, 1.0, 2.0],
    [-_INF, 0.0, 1.0],
    [-_INF, -_INF, -_INF],
    [_INF, 0.0, 0.0],
    [_NAN, 0.0, 0.0],
]

_CONCENTRATION_VECTORS: list[list[float]] = [
    [1.0, 2.0, 3.0],
    [0.5, 0.5, 0.5],
    [0.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [_INF, 1.0, 1.0],
    [_NAN, 1.0, 1.0],
]

_COVARIANCES: list[list[list[float]]] = [
    [[2.0, 0.5], [0.5, 1.0]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[1.0, 2.0], [2.0, 1.0]],
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0]],
    [[_NAN, 0.0], [0.0, 1.0]],
]

_SCALE_TRILS: list[list[list[float]]] = [
    [[1.0, 0.0], [0.5, 2.0]],
    [[1.0, 0.3], [0.5, 2.0]],
    [[-1.0, 0.0], [0.5, 2.0]],
    [[0.0, 0.0], [0.5, 1.0]],
]

_SCALAR_VALUES: list[float] = [
    -_INF,
    -1.0,
    -0.5,
    0.0,
    0.3,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    7.0,
    _INF,
    _NAN,
]

_VECTOR_VALUES: list[list[float]] = _PROB_VECTORS + [[0.0, 1.0, 0.0]]

_COUNT_VALUES: list[list[float]] = [
    [1.0, 2.0, 2.0],
    [0.0, 0.0, 5.0],
    [1.0, 1.0, 1.0],
    [3.0, 3.0, 0.0],
    [-1.0, 3.0, 3.0],
    [1.5, 1.5, 2.0],
]

_MATRIX_VALUES: list[list[list[float]]] = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[2.0, 0.5], [0.5, 1.0]],
    [[1.0, 2.0], [2.0, 1.0]],
    [[1.0, 0.0], [0.6, 0.8]],
    [[1.0, 0.0], [0.5, 0.5]],
    [[1.0, 0.6], [0.0, 0.8]],
    [[1.0, 0.0], [-0.6, -0.8]],
    [[0.0, 0.0], [0.0, 0.0]],
]

_EVENT_VALUES: list[list[float]] = [[0.0, 1.0, 2.0], [-1.0, 0.0, 0.0], [_NAN, 0.0, 0.0]]

# ``build(M, t, validate_args, **override)`` constructs a distribution in the
# library ``M`` (``lucid.distributions`` or the reference's), with ``t``
# turning a nested list into that library's tensor and ``override``
# replacing one nominal parameter.
_Builder = Callable[..., Any]


class _Case:
    """How to build one distribution, what to offer each parameter, and
    which values to offer its support."""

    def __init__(
        self, build: _Builder, params: dict[str, list[Any]], values: list[Any]
    ) -> None:
        self.build = build
        self.params = params
        self.values = values


def _plain(
    name: str, nominal: dict[str, Any], values: list[Any] | None = None
) -> _Case:
    """A distribution whose parameters are all scalars passed by keyword."""

    def build(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
        kw = {k: t(o.get(k, v)) for k, v in nominal.items()}
        return getattr(M, name)(**kw, validate_args=va)

    return _Case(build, {k: _SCALARS for k in nominal}, values or _SCALAR_VALUES)


def _dual(
    name: str,
    nominal: Any,
    extra: dict[str, Any],
    params: dict[str, list[Any]],
    values: list[Any],
) -> _Case:
    """A distribution taking ``probs`` or ``logits``, plus other parameters."""

    def build(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
        kw = {
            k: (v if k == "total_count" and isinstance(v, int) else t(v))
            for k, v in (extra | {k: v for k, v in o.items() if k in extra}).items()
        }
        if "logits" in o:
            kw["logits"] = t(o["logits"])
        else:
            kw["probs"] = t(o.get("probs", nominal))
        return getattr(M, name)(**kw, validate_args=va)

    return _Case(build, params, values)


def _binary(name: str, extra: dict[str, Any] | None = None) -> _Case:
    """``probs`` / ``logits`` scalars, as ``Bernoulli``."""
    extra = extra or {}
    params = {"probs": _SCALARS, "logits": _SCALARS} | {k: _SCALARS for k in extra}
    return _dual(name, 0.3, extra, params, _SCALAR_VALUES)


def _categorical(name: str, extra: dict[str, Any] | None = None) -> _Case:
    """``probs`` / ``logits`` vectors, as ``Categorical``."""
    extra = extra or {}
    params: dict[str, list[Any]] = {"probs": _PROB_VECTORS, "logits": _LOGIT_VECTORS}
    params |= {k: _SCALARS for k in extra if k != "total_count"}
    return _dual(name, [0.2, 0.3, 0.5], extra, params, _VECTOR_VALUES)


def _dirichlet(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    return M.Dirichlet(t(o.get("concentration", [1.0, 2.0, 3.0])), validate_args=va)


def _mvn(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    loc = t(o.get("loc", [0.0, 0.0]))
    for key in ("precision_matrix", "scale_tril"):
        if key in o:
            return M.MultivariateNormal(loc, **{key: t(o[key])}, validate_args=va)
    cov = t(o.get("covariance_matrix", [[2.0, 0.5], [0.5, 1.0]]))
    return M.MultivariateNormal(loc, covariance_matrix=cov, validate_args=va)


def _wishart(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    df = t(o.get("df", 3.0))
    for key in ("precision_matrix", "scale_tril"):
        if key in o:
            return M.Wishart(df, **{key: t(o[key])}, validate_args=va)
    cov = t(o.get("covariance_matrix", [[2.0, 0.5], [0.5, 1.0]]))
    return M.Wishart(df, covariance_matrix=cov, validate_args=va)


def _lkj(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    return M.LKJCholesky(2, t(o.get("concentration", 1.5)), validate_args=va)


def _independent(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    base = M.Normal(t([0.0, 1.0, 2.0]), t([1.0, 1.0, 2.0]))
    return M.Independent(base, 1, validate_args=va)


def _transformed(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    base = M.Normal(t(0.0), t(1.0))
    return M.TransformedDistribution(
        base, [M.transforms.ExpTransform()], validate_args=va
    )


def _mixture(M: Any, t: Callable[[Any], Any], va: bool | None, **o: Any) -> Any:
    return M.MixtureSameFamily(
        M.Categorical(probs=t([0.3, 0.7])),
        M.Normal(t([0.0, 1.0]), t([1.0, 2.0])),
        validate_args=va,
    )


_MATRIX_PARAMS: dict[str, list[Any]] = {
    "covariance_matrix": _COVARIANCES,
    "precision_matrix": _COVARIANCES,
    "scale_tril": _SCALE_TRILS,
}

CASES: dict[str, _Case] = {
    "Normal": _plain("Normal", {"loc": 0.5, "scale": 1.5}),
    "LogNormal": _plain("LogNormal", {"loc": 0.5, "scale": 1.5}),
    "Uniform": _plain("Uniform", {"low": 0.0, "high": 2.0}),
    "Exponential": _plain("Exponential", {"rate": 1.5}),
    "Laplace": _plain("Laplace", {"loc": 0.5, "scale": 1.5}),
    "Cauchy": _plain("Cauchy", {"loc": 0.5, "scale": 1.5}),
    "StudentT": _plain("StudentT", {"df": 3.0, "loc": 0.5, "scale": 1.5}),
    "Pareto": _plain("Pareto", {"scale": 1.0, "alpha": 2.0}),
    "Weibull": _plain("Weibull", {"scale": 1.0, "concentration": 1.5}),
    "HalfNormal": _plain("HalfNormal", {"scale": 1.5}),
    "HalfCauchy": _plain("HalfCauchy", {"scale": 1.5}),
    "FisherSnedecor": _plain("FisherSnedecor", {"df1": 3.0, "df2": 5.0}),
    "Gamma": _plain("Gamma", {"concentration": 2.0, "rate": 1.5}),
    "Chi2": _plain("Chi2", {"df": 3.0}),
    "Beta": _plain("Beta", {"concentration1": 2.0, "concentration0": 3.0}),
    "Gumbel": _plain("Gumbel", {"loc": 0.5, "scale": 1.5}),
    "InverseGamma": _plain("InverseGamma", {"concentration": 2.0, "rate": 1.5}),
    "Kumaraswamy": _plain(
        "Kumaraswamy", {"concentration1": 2.0, "concentration0": 3.0}
    ),
    "Poisson": _plain("Poisson", {"rate": 1.5}),
    "Dirichlet": _Case(
        _dirichlet, {"concentration": _CONCENTRATION_VECTORS}, _VECTOR_VALUES
    ),
    "Bernoulli": _binary("Bernoulli"),
    "Geometric": _binary("Geometric"),
    "ContinuousBernoulli": _binary("ContinuousBernoulli"),
    "Binomial": _binary("Binomial", {"total_count": 5.0}),
    "NegativeBinomial": _binary("NegativeBinomial", {"total_count": 3.0}),
    "RelaxedBernoulli": _binary("RelaxedBernoulli", {"temperature": 0.5}),
    "Categorical": _categorical("Categorical"),
    "OneHotCategorical": _categorical("OneHotCategorical"),
    "RelaxedOneHotCategorical": _categorical(
        "RelaxedOneHotCategorical", {"temperature": 0.5}
    ),
    "Multinomial": _categorical("Multinomial", {"total_count": 5}),
    "MultivariateNormal": _Case(
        _mvn,
        {"loc": [[0.0, 0.0], [_INF, 0.0], [_NAN, 0.0]]} | _MATRIX_PARAMS,
        [[0.0, 1.0]],
    ),
    "Wishart": _Case(_wishart, {"df": _SCALARS} | _MATRIX_PARAMS, _MATRIX_VALUES),
    "LKJCholesky": _Case(_lkj, {"concentration": _SCALARS}, _MATRIX_VALUES),
    "Independent": _Case(_independent, {}, _EVENT_VALUES),
    "TransformedDistribution": _Case(_transformed, {}, _SCALAR_VALUES),
    "MixtureSameFamily": _Case(_mixture, {}, _SCALAR_VALUES),
}
CASES["Multinomial"].values = _COUNT_VALUES

# Every difference from the reference that is kept on purpose, keyed by
# (distribution, aspect) — the aspect is ``param:<name>``, ``support`` or
# ``table`` — with the reason.
_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("Multinomial", "param:probs"): "the parameter is held privately, so "
    "validation never reaches it — any vector constructs",
    ("Multinomial", "param:logits"): "as for probs",
    ("Multinomial", "support"): "element-wise whole counts, where the reference "
    "admits any non-negative vector summing to at most total_count",
    ("Multinomial", "table"): "the table holds total_count and probs, and the "
    "support's event_dim is 0",
    ("MultivariateNormal", "param:scale_tril"): "only loc is in the table; a "
    "scale_tril is taken as given",
    ("MultivariateNormal", "table"): "only loc is in the table, and the support "
    "is the element-wise real line",
    ("Wishart", "param:scale_tril"): "only df is in the table; a scale_tril is "
    "taken as given",
    ("Wishart", "table"): "only df is in the table",
    ("RelaxedBernoulli", "param:temperature"): "the reference does not validate "
    "it; a non-positive temperature divides the logits by zero or flips them",
    ("RelaxedBernoulli", "table"): "temperature is in the table",
    ("RelaxedOneHotCategorical", "param:temperature"): "as RelaxedBernoulli",
    ("RelaxedOneHotCategorical", "table"): "temperature is in the table",
    ("TransformedDistribution", "support"): "Lucid transforms declare no "
    "codomain, so there is no support to report",
    ("TransformedDistribution", "table"): "the support is None",
}


def _distribution_names() -> list[str]:
    return sorted(
        n
        for n in D.__all__
        if isinstance(getattr(D, n), type)
        and issubclass(getattr(D, n), D.Distribution)
        and n not in ("Distribution", "ExponentialFamily")
    )


def _lucid_tensor(v: Any) -> lucid.Tensor:
    return lucid.tensor(np.asarray(v, dtype=np.float32))


def _ref_tensor(ref: Any) -> Callable[[Any], Any]:
    return lambda v: ref.tensor(np.asarray(v, dtype=np.float32))


def _accepts(build: Callable[[], Any]) -> bool:
    try:
        build()
    except Exception:  # noqa: BLE001 — any refusal counts
        return False
    return True


def _admits(constraint: Any, value: Any) -> bool:
    try:
        return bool(constraint.check(value).all().item())
    except Exception:  # noqa: BLE001 — a check that cannot run admits nothing
        return False


def _judge(name: str, aspect: str, differences: list[Any]) -> None:
    reason = _EXCEPTIONS.get((name, aspect))
    if reason is None:
        assert (
            not differences
        ), f"{name} {aspect} differs from the reference: {differences}"
    else:
        assert differences, (
            f"{name} {aspect} now matches the reference; remove its entry from "
            f"_EXCEPTIONS ({reason})"
        )


def test_every_distribution_has_a_case() -> None:
    assert sorted(CASES) == _distribution_names()


def test_every_exception_names_a_case() -> None:
    for name, aspect in _EXCEPTIONS:
        assert name in CASES, name
        kind, _, param = aspect.partition(":")
        assert kind in ("param", "support", "table"), aspect
        if kind == "param":
            assert param in CASES[name].params, aspect


_PARAMS = [(n, p) for n in sorted(CASES) for p in CASES[n].params]


@pytest.mark.parametrize("name,param", _PARAMS, ids=[f"{n}.{p}" for n, p in _PARAMS])
def test_parameter_domain(name: str, param: str, ref: Any) -> None:
    case = CASES[name]
    rt = _ref_tensor(ref)
    differences = []
    for v in case.params[param]:
        ours = _accepts(lambda: case.build(D, _lucid_tensor, None, **{param: v}))
        theirs = _accepts(lambda: case.build(ref.distributions, rt, True, **{param: v}))
        if ours != theirs:
            differences.append((v, "Lucid accepts" if ours else "Lucid refuses"))
    _judge(name, f"param:{param}", differences)


@pytest.mark.parametrize("name", sorted(CASES))
def test_support(name: str, ref: Any) -> None:
    case = CASES[name]
    rt = _ref_tensor(ref)
    ours = case.build(D, _lucid_tensor, None).support
    theirs = case.build(ref.distributions, rt, True).support
    differences = []
    for v in case.values:
        a = ours is not None and _admits(ours, _lucid_tensor(v))
        b = _admits(theirs, rt(v))
        if a != b:
            differences.append((v, "Lucid admits" if a else "Lucid refuses"))
    _judge(name, "support", differences)


@pytest.mark.parametrize("name", sorted(CASES))
def test_constraint_table(name: str, ref: Any) -> None:
    case = CASES[name]
    ours = case.build(D, _lucid_tensor, None)
    theirs = case.build(ref.distributions, _ref_tensor(ref), True)
    differences: list[Any] = []
    if set(ours.arg_constraints) != set(theirs.arg_constraints):
        differences.append(("arg_constraints", sorted(ours.arg_constraints)))
    if ours.support is None:
        differences.append(("support", None))
    else:
        for attr in ("is_discrete", "event_dim"):
            if getattr(ours.support, attr) != getattr(theirs.support, attr):
                differences.append((attr, getattr(ours.support, attr)))
    _judge(name, "table", differences)
