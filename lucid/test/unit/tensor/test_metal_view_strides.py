"""A Metal view's strides are worked out from its op, not replayed on the CPU.

``view`` and ``reshape`` of a traced Metal view — a transpose, a permute, a
slice — need the strides the same view would have on the CPU, to know
whether the new shape can read it in place.  They were read back by
replaying the view on a CPU tensor: a host allocation and a CPU op for every
merge-heads ``attn.permute(0, 2, 1, 3).reshape(b, n, e)``, in every attention
layer, every step (LCD-462).  :mod:`lucid._tensor._metal_views` now works
them out from the op's arguments; these tests hold each rule to the CPU and
to zero host work.

Strides are compared on the axes that move an element.  An axis of length
one moves none, and the CPU's choice of stride for it depends on how the
view was composed (a slice squeezes its window axis, which re-strides every
length-one axis), so it says nothing about the layout.
"""

import random
from collections.abc import Callable

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._tensor import _metal_views
from lucid.test._fixtures.devices import metal_available

pytestmark = pytest.mark.skipif(not metal_available(), reason="metal unavailable")

_CPU = _C_engine.Device.CPU

type _View = Callable[[lucid.Tensor], lucid.Tensor]


def _moving(
    shape: tuple[int, ...], stride: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    """The strides of the axes that move an element; ``()`` when none can."""
    assert stride is not None, f"no strides for a view of shape {shape}"
    if 0 in shape:
        return ()
    return tuple(st for n, st in zip(shape, stride, strict=True) if n != 1)


def _base(device: str) -> lucid.Tensor:
    # A traced view itself, so the rule under test reads its input's strides
    # from another rule.
    return lucid.arange(120.0).reshape(2, 3, 4, 5).to(device).permute(3, 1, 0, 2)


#: One view per rule, and per spelling of the ones a free function shares.
#: Every one is taken of ``_base`` — shape (5, 3, 2, 4).
VIEWS: dict[str, _View] = {
    "transpose": lambda t: t.transpose(0, 2),
    "transpose-kwargs": lambda t: t.transpose(dim0=-1, dim1=1),
    "transpose-all": lambda t: t.transpose(),
    "lucid.transpose": lambda t: lucid.transpose(t, 0, 2),
    "swapaxes": lambda t: t.swapaxes(1, -1),
    "permute-spread": lambda t: t.permute(2, 0, 3, 1),
    "permute-tuple": lambda t: t.permute((2, 0, 3, 1)),
    "permute-kwargs": lambda t: t.permute(dims=[2, 0, 3, 1]),
    "lucid.permute": lambda t: lucid.permute(t, (2, 0, 3, -3)),
    "movedim": lambda t: t.movedim(0, -1),
    "movedim-lists": lambda t: t.movedim((0, 1), (3, 2)),
    "lucid.movedim": lambda t: lucid.movedim(t, source=[1], destination=[0]),
    "t": lambda t: t[0, 0].t(),
    "T": lambda t: t.T,
    "mT": lambda t: t.mT,
    "narrow": lambda t: t.narrow(0, 1, 3),
    "diagonal": lambda t: t.diagonal(),
    "diagonal-dims": lambda t: t.diagonal(1, 0, 2),
    "lucid.diagonal": lambda t: lucid.diagonal(t, offset=-1, dim1=3, dim2=0),
    "unfold": lambda t: t.unfold(0, 2, 2),
    "unfold-last": lambda t: t.unfold(-1, 3, 1),
    "expand-leading": lambda t: t.expand(2, 5, 3, 2, 4),
    "expand-stretch": lambda t: t[:, :1].expand(-1, 3, -1, -1),
    "broadcast_to": lambda t: t[:1].broadcast_to((3, 5, 3, 2, 4)),
    "expand_as": lambda t: t[:, :, :1].expand_as(t),
    "as_strided": lambda t: t.as_strided((2, 3), (4, 1)),
    "split": lambda t: t.split(2, dim=0)[1],
    "lucid.split": lambda t: lucid.split(t, [1, 3], dim=3)[1],
    "chunk": lambda t: t.chunk(2, dim=3)[1],
    "unbind": lambda t: t.unbind(1)[2],
    "lucid.unbind": lambda t: lucid.unbind(t, dim=-1)[0],
    "index-int": lambda t: t[1],
    "index-step": lambda t: t[:, ::2],
    "index-mixed": lambda t: t[1:, 0, ::3],
    "index-none": lambda t: t[1:, None, ..., 2],
    "index-none-inner": lambda t: t[..., None],
    "index-empty": lambda t: t[3:1],
}


def test_every_replayed_view_op_has_a_stride_rule() -> None:
    replayed = {
        *_metal_views._REARRANGEMENTS,
        *_metal_views._OVERLAPPING,
        *_metal_views._PIECES,
        "__getitem__",
        "T",
        "mT",
    }
    assert set(_metal_views._STRIDE_RULES) == replayed


@pytest.mark.parametrize("make", list(VIEWS.values()), ids=list(VIEWS))
def test_a_metal_view_stride_dispatches_nothing_and_is_the_cpus(make: _View) -> None:
    view = make(_base("metal"))
    host = _C_engine.memory_stats(_CPU).alloc_count
    with lucid.profiler.profile() as prof:
        stride = _metal_views._stride(view)
    assert [event.name for event in prof.events()] == []
    assert _C_engine.memory_stats(_CPU).alloc_count == host

    cpu = make(_base("cpu"))
    assert view.shape == cpu.shape
    assert _moving(view.shape, stride) == _moving(cpu.shape, cpu.stride())


def test_merging_heads_on_metal_runs_no_cpu_op() -> None:
    # MultiheadAttention's merge: the step-cost budget's transformer cell.
    x = lucid.randn(2, 8, 32, device="metal")
    heads = x.reshape(2, 8, 4, 8).permute(0, 2, 1, 3)
    host = _C_engine.memory_stats(_CPU).alloc_count
    with lucid.profiler.profile() as prof:
        merged = heads.permute(0, 2, 1, 3).reshape(2, 8, 32)
    assert [event.name for event in prof.events()] == ["permute", "reshape"]
    assert _C_engine.memory_stats(_CPU).alloc_count == host
    assert merged.tolist() == x.tolist()


def test_a_reversed_slice_is_still_no_view_to_reshape_through() -> None:
    # The CPU gathers ``x[::-1]`` into a copy, so its Metal stand-in has no
    # strides a later ``view`` could step through — as before the rules.  An
    # empty one is a zero-length cut of ``x`` on the CPU, which any shape views.
    x = lucid.arange(6.0, device="metal")
    assert _metal_views._stride(x[::-1]) is None
    assert _metal_views._stride(x[1:4:-1]) is not None


# ── Random chains, held to the CPU ───────────────────────────────────────────


def _shape_of(rng: random.Random, numel: int) -> tuple[int, ...]:
    """A random shape with ``numel`` elements, length-one axes included."""
    dims: list[int] = []
    left = numel
    while left > 1 and len(dims) < 4:
        size = rng.choice([d for d in range(2, left + 1) if left % d == 0])
        dims.append(size)
        left //= size
    if left > 1:
        dims.append(left)
    for _ in range(rng.randrange(2)):
        dims.insert(rng.randrange(len(dims) + 1), 1)
    rng.shuffle(dims)
    return tuple(dims)


def _key(rng: random.Random, shape: tuple[int, ...]) -> tuple[object, ...]:
    parts: list[object] = []
    for n in shape[: rng.randrange(len(shape) + 1)]:
        pick = rng.randrange(4)
        if pick == 0 and n:
            parts.append(rng.randrange(-n, n))
        elif pick == 1:
            start = rng.randrange(n + 1)
            parts.append(slice(start, rng.randrange(start, n + 1), rng.randrange(1, 4)))
        else:
            parts.append(slice(None))
        if rng.randrange(5) == 0:
            parts.append(None)
    if rng.randrange(3) == 0:
        parts.append(Ellipsis)
    return tuple(parts)


def _ranked_views(rng: random.Random, shape: tuple[int, ...]) -> list[_View]:
    n = len(shape)
    perm = rng.sample(range(n), n)
    i, j = rng.randrange(n), rng.randrange(n)
    size = rng.randrange(1, shape[i] + 1) if shape[i] else 0
    start = rng.randrange(shape[i] - size + 1) if shape[i] else 0
    step = rng.randrange(1, 4)
    piece = rng.randrange(max(shape[i], 1))
    stretch = tuple(rng.choice([2, 3]) if s == 1 else -1 for s in shape)
    views: list[_View] = [
        lambda t: t.permute(*perm),
        lambda t: t.transpose(i, j),
        lambda t: t.movedim(i, j),
        lambda t: t.narrow(i, start, size),
        lambda t: t.unbind(i)[piece] if shape[i] else t,
        lambda t: t.split(max(size, 1), dim=i)[0],
        lambda t: t.expand(*stretch),
        lambda t: t.expand(2, *shape),
        lambda t: t.T,
    ]
    if shape[i]:
        views.append(lambda t: t.unfold(i, size, step))
    if n >= 2 and i != j:
        offset = rng.randrange(-2, 3)
        views += [lambda t: t.diagonal(offset, i, j), lambda t: t.mT]
    return views


def _views(rng: random.Random, shape: tuple[int, ...]) -> list[_View]:
    """Every view op the random chains draw from, with arguments for ``shape``."""
    numel = 1
    for n in shape:
        numel *= n
    target = _shape_of(rng, numel) if numel else (0, *shape)
    key = _key(rng, shape)
    views: list[_View] = [
        lambda t: t[key],
        lambda t: t.reshape(*target),
        lambda t: t.unsqueeze(rng.randrange(len(shape) + 1)),
        lambda t: t.squeeze(),
        lambda t: t.detach(),
    ]
    if shape:
        views += _ranked_views(rng, shape)
    return views


def _same_view(cpu: lucid.Tensor, metal: lucid.Tensor, where: str) -> None:
    assert metal.shape == cpu.shape, where
    got = _moving(metal.shape, _metal_views._stride(metal))
    assert got == _moving(cpu.shape, cpu.stride()), where
    assert metal.tolist() == cpu.tolist(), where


def _attempt(view: _View, t: lucid.Tensor) -> lucid.Tensor | None:
    try:
        return view(t)
    except RuntimeError:
        return None


@pytest.mark.parametrize("seed", range(24))
def test_random_view_chains_have_the_cpus_strides(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(12):
        numel = rng.choice([1, 6, 8, 12, 24, 36, 48])
        shape = _shape_of(rng, numel)
        cpu = lucid.arange(float(numel)).reshape(shape)
        metal = cpu.to("metal")
        steps = [f"{shape}"]
        for _ in range(rng.randrange(1, 5)):
            view = rng.choice(_views(rng, tuple(cpu.shape)))
            # Draw once and replay the same op on both devices.
            state = rng.getstate()
            next_cpu = _attempt(view, cpu)
            rng.setstate(state)
            next_metal = _attempt(view, metal)
            assert (next_cpu is None) == (next_metal is None), steps
            if next_cpu is None or next_metal is None:
                continue
            cpu, metal = next_cpu, next_metal
            steps.append(f"-> {tuple(cpu.shape)}")
            _same_view(cpu, metal, " ".join(steps))
        # ``view`` refuses on Metal exactly where it refuses on the CPU.
        target = _shape_of(rng, cpu.numel()) if cpu.numel() else (0,)
        cpu_view = _attempt(lambda t: t.view(*target), cpu)
        metal_view = _attempt(lambda t: t.view(*target), metal)
        assert (cpu_view is None) == (metal_view is None), (steps, target)
        if cpu_view is not None and metal_view is not None:
            _same_view(cpu_view, metal_view, f"{steps} view {target}")


# ── The Python chunk rule is the engine's ────────────────────────────────────


def _layout(rng: random.Random) -> lucid.Tensor:
    """A CPU view of random shape and strides over a buffer large enough for it."""
    shape = [rng.choice([1, 1, 2, 3, 4]) for _ in range(rng.randrange(5))]
    if shape and rng.randrange(8) == 0:
        shape[rng.randrange(len(shape))] = 0
    stride = [rng.choice([0, 1, 2, 3, 4, 6, 8, 12, 24]) for _ in shape]
    span = 1 + sum((n - 1) * st for n, st in zip(shape, stride, strict=True) if n)
    buffer = lucid.zeros(span)
    return _wrap(_C_engine.TensorImpl._make_view(buffer._impl, shape, stride, 0))


@pytest.mark.parametrize("seed", range(16))
def test_the_python_view_rule_is_the_engines(seed: int) -> None:
    rng = random.Random(1000 + seed)
    for _ in range(40):
        a = _layout(rng)
        shape, stride = tuple(a.shape), tuple(a.stride())
        target = _shape_of(rng, a.numel()) if a.numel() else (0, *shape)
        try:
            engine: tuple[int, ...] | None = tuple(a.view(*target).stride())
        except RuntimeError:
            engine = None
        python = _metal_views._view_strides(shape, stride, target, False)
        assert (python is None) == (engine is None), (shape, stride, target)
        if python is not None:
            assert _moving(target, python) == _moving(target, engine)
        # Dropping and inserting length-one axes is always a view, and the
        # engine strides every length-one axis the same way.
        squeezed = a.squeeze()
        assert _metal_views._view_strides(
            shape, stride, tuple(squeezed.shape), True
        ) == tuple(squeezed.stride())
        unsqueezed = a.unsqueeze(rng.randrange(len(shape) + 1))
        assert _metal_views._view_strides(
            shape, stride, tuple(unsqueezed.shape), True
        ) == tuple(unsqueezed.stride())
