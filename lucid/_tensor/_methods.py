"""
Auto-inject Tensor methods from the ops registry.

Every method is generated from an ``OpEntry`` in ``_registry.py``; signature
translation (``dim=None|int|list``, ``keepdim``, ``correction`` ...) lives
in ``_adapters.py``.  Only ``eval`` — which calls into the impl handle
directly and isn't an engine op — is special-cased below.

Called once at module import time by ``tensor.py``.
"""

from typing import TYPE_CHECKING, cast

from lucid._dispatch import _unwrap, _wrap
from lucid._tensor._indexing import _adopt_inplace
from lucid._ops._registry import _REGISTRY, OpEntry, _bind_leading

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


_OPERANDS: dict[str, tuple[str, ...]] = {}


def _operand_names(e: OpEntry) -> tuple[str, ...]:
    """The advertised names of ``e``'s tensor operands after ``self``.

    Read from the free function's signature, so the method and
    ``lucid.<op>`` bind the same keywords — once per op, on its first
    keyword call: ``lucid._ops`` imports this package, so it cannot be
    imported while the methods are made.
    """
    names = _OPERANDS.get(e.name)
    if names is None:
        from lucid._ops import _leading_tensor_names

        names = _OPERANDS[e.name] = _leading_tensor_names(e)[1:]
    return names


def _fixed_arity_method(e: OpEntry, tensor_cls: type) -> object:
    """The Tensor method for ``e`` when ``self`` is followed by fixed arguments.

    Unwraps the leading ``Tensor`` positional args (up to the op's declared
    tensor-arg count), dispatches to the engine op, and either wraps the
    result as a Tensor or returns the raw value per ``e.returns_tensor``.
    """
    # ``self`` is the only operand of a one-tensor op: nothing to bind.
    binds = e.n_tensor_args > 1

    def method(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        """Standard-arity method body for registry-driven Tensor methods."""
        if kwargs and binds:
            names = _operand_names(e)
            args, kwargs = _bind_leading(method.__name__, names, args, kwargs)
        proc_args: list[object] = []
        for i, a in enumerate(args):
            if i < (e.n_tensor_args - 1) and isinstance(a, tensor_cls):
                proc_args.append(_unwrap(a))  # type: ignore[arg-type]
            else:
                proc_args.append(a)

        result = e.engine_fn(self._impl, *proc_args, **kwargs)
        if e.inplace:
            return _adopt_inplace(self, result, e.name)
        if e.returns_tensor:
            if isinstance(result, (list, tuple)):
                return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
            return _wrap(result)
        return cast("Tensor", result)

    method.__name__ = e.method_name or e.name
    return method


def _inject_methods(tensor_cls: type) -> None:
    """Attach all registry ops as Tensor methods."""
    # The generated methods are annotated ``self: Tensor`` and
    # ``-> Tensor``, and ``Tensor`` is imported here for type checking
    # only, so under PEP 649 nothing could resolve those annotations at
    # runtime: ``inspect.signature`` raised NameError on 193 of the 236
    # callables on Tensor, and ``typing.get_type_hints`` with them.
    # Everything that reads annotations — an IDE, ``help()``, a docs
    # walker — saw the same wall.
    #
    # The class is right here as ``tensor_cls``.  Binding the name it is
    # already written as costs nothing and cannot drift, because the
    # TYPE_CHECKING import and this binding name the same object.
    globals().setdefault("Tensor", tensor_cls)

    def _make_method(e: OpEntry) -> object:
        """Build a Tensor method that dispatches to ``e.engine_fn``.

        Two branches are emitted based on ``e.n_tensor_args``: a variadic
        version (``-1``) that accepts a list of tensors after ``self``, and
        the standard version that takes a fixed-arity argument list.
        """
        if e.n_tensor_args == -1:

            def method_list(
                self: Tensor,
                tensors: list[Tensor],
                *args: object,
                **kwargs: object,
            ) -> Tensor:
                """Variadic-tensor method body — prepends ``self`` to the unwrapped tensor list."""
                all_tensors = [_unwrap(t) for t in [self] + list(tensors)]
                result = e.engine_fn(all_tensors, *args, **kwargs)
                if e.returns_tensor:
                    if isinstance(result, (list, tuple)):
                        return type(result)(_wrap(r) for r in result)  # type: ignore[return-value]
                    return _wrap(result)
                return cast("Tensor", result)

            method_list.__name__ = e.method_name or e.name
            return method_list
        return _fixed_arity_method(e, tensor_cls)

    for entry in _REGISTRY:
        if entry.method_name is None:
            continue
        # skip dunders — handled in _dunders.py
        if entry.method_name.startswith("__"):
            continue
        setattr(tensor_cls, entry.method_name, _make_method(entry))

    # ── methods that bypass the registry — only those with no clean
    # registry mapping live here.  Everything else is auto-injected from
    # ``_REGISTRY`` above (now that ``_make_method`` forwards keyword args
    # the registry adapters handle the API-compat translation directly).

    def eval(self: Tensor) -> Tensor:
        """Force immediate evaluation of this tensor.

        Special-cased because it operates on the impl handle rather than going
        through any engine op — so it has no place in the ops registry.
        On Metal this flushes the lazy compute graph; on CPU it's a no-op.
        Returns ``self`` for chaining.
        """
        self._impl.eval()
        return self

    setattr(tensor_cls, "eval", eval)
