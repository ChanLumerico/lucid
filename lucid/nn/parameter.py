"""
nn.Parameter: a Tensor that is automatically registered by Module.
"""

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._factories.converters import _to_impl


class Parameter(Tensor):
    r"""A :class:`Tensor` subclass that :class:`Module` recognises as a learnable parameter.

    When an instance is bound as an attribute of a ``Module``, the module's
    ``__setattr__`` hook detects the ``_is_parameter`` marker and routes it
    into the module's ``_parameters`` registry instead of treating it as a
    plain Tensor attribute.  This is what makes ``Module.parameters()`` and
    ``Module.named_parameters()`` discover the tensor automatically — no
    manual registration call is needed.

    By default the underlying tensor has ``requires_grad=True``, so it
    participates in the autograd graph from the moment of construction.
    Pass ``requires_grad=False`` for "buffer-like" learnables (e.g.
    BatchNorm's running mean) that should still be tracked under
    ``state_dict`` but shouldn't accumulate gradients.

    Parameters
    ----------
    data : Tensor | list | None, optional
        Initial values.  ``None`` (default) yields a zero-sized
        placeholder F32/CPU parameter, useful when the actual shape is
        deferred until first ``forward`` (lazy modules).  A ``Tensor``
        clones its storage so the new parameter is independent of the
        source.  A list / nested-list is converted via :func:`_to_impl`.
    requires_grad : bool, optional
        Whether to record gradient flow through this parameter
        (default: ``True``).

    Attributes
    ----------
    _is_parameter : bool
        Class-level marker (always ``True``) consumed by ``Module``'s
        attribute-binding logic.

    Notes
    -----
    Subclassing ``Tensor`` (rather than wrapping one) keeps every tensor
    operation usable on a ``Parameter`` without unwrap boilerplate — e.g.
    ``p + 1`` returns a ``Tensor``, while ``p.data`` and ``p.grad`` still
    work.  The ``_is_parameter`` class attribute is the sole behavioural
    difference visible to ``Module``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn import Parameter, Module
    >>> class Affine(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.w = Parameter(lucid.randn(3))  # auto-registered
    ...         self.b = Parameter(lucid.zeros(3))
    >>> m = Affine()
    >>> list(m.named_parameters())   # both surface here automatically
    [('w', Parameter ...), ('b', Parameter ...)]
    """

    _is_parameter: bool = True

    def __new__(
        cls,
        data: Tensor | list[object] | None = None,
        requires_grad: bool = True,
    ) -> Parameter:
        """Allocate a new :class:`Parameter` instance.

        Bypasses :meth:`Tensor.__init__` (which expects raw data) so a
        Parameter can be cloned from an existing Tensor without an extra
        copy of metadata.  Always clones the input storage so the
        Parameter owns an independent buffer.

        Parameters
        ----------
        data : Tensor | list | None, optional
            See class docstring.
        requires_grad : bool, optional
            See class docstring.

        Returns
        -------
        Parameter
            Fresh instance ready to be bound as a Module attribute.
        """
        if data is None:
            impl = _C_engine.zeros([0], _C_engine.F32, _C_engine.CPU)
            impl = impl.clone_with_grad(requires_grad)
        elif isinstance(data, Tensor):
            impl = data._impl.clone_with_grad(requires_grad)
        else:
            impl = _to_impl(data, requires_grad=requires_grad)
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    def __init__(
        self,
        data: Tensor | list[object] | None = None,
        requires_grad: bool = True,
    ) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        pass

    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return f"Parameter containing:\n{super().__repr__()}"
