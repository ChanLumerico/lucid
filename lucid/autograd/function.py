"""
autograd.Function: base class for custom differentiable operations.
"""

from typing import Any, Protocol, cast, override
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._tensor.tensor import Tensor
from lucid.autograd._python_node import _register

#: A saved tensor's impl, its version when ``forward`` returned, and where it
#: came from ("Mul input 1") for the ``VersionMismatch`` message.
type _SavedEntry = tuple[_C_engine.TensorImpl, int, str]


def _impl_of(t: object) -> _C_engine.TensorImpl | None:
    """The impl behind a saved entry, or ``None`` for one that is not a tensor."""
    if isinstance(t, _C_engine.TensorImpl):
        return t
    if isinstance(t, Tensor):
        return t._impl
    return None


def _position(
    impl: _C_engine.TensorImpl, impls: list[_C_engine.TensorImpl | None]
) -> int | None:
    """Index of ``impl`` in ``impls`` by identity, or ``None``."""
    for k, other in enumerate(impls):
        if other is impl:
            return k
    return None


class _FunctionClass(Protocol):
    """Protocol describing a Function subclass (has forward/backward classmethods)."""

    @classmethod
    def forward(
        cls, ctx: object, *args: object, **kwargs: object
    ) -> Tensor | tuple[Tensor, ...]: ...

    @classmethod
    def backward(
        cls, ctx: object, *grad_outputs: Tensor
    ) -> Tensor | tuple[Tensor | None, ...] | list[Tensor | None]: ...


class FunctionCtx:
    r"""Per-call context shared between :meth:`Function.forward` and
    :meth:`Function.backward`.

    A fresh ``FunctionCtx`` is created on every :meth:`Function.apply`
    invocation. ``forward`` populates it with anything ``backward``
    will need: saved tensors (via :meth:`save_for_backward`),
    non-differentiable output markers (via
    :meth:`mark_non_differentiable`), and arbitrary user-defined
    attributes (cached shapes, axis indices, scalar hyperparameters,
    ...) set with ordinary attribute assignment. The context is the
    only legal channel for passing state from forward to backward —
    capturing tensors through Python closures bypasses autograd's
    bookkeeping and leaks memory.

    Parameters
    ----------
    None
        ``FunctionCtx`` is instantiated by :meth:`Function.apply`
        with no arguments. User code never constructs one
        directly; it receives the instance as the first
        positional argument of ``forward`` / ``backward``.

    Attributes
    ----------
    needs_input_grad : tuple of bool
        One flag per positional ``Tensor`` input to ``forward``,
        indicating whether autograd would propagate a gradient to
        that input. Use this to skip unneeded branches in
        ``backward``.
    saved_tensors : tuple of Tensor
        Read-only view of the tensors stored via
        :meth:`save_for_backward`, in registration order.

    Methods
    -------
    save_for_backward(\*tensors)
        Persist tensors needed for the backward pass.
    mark_non_differentiable(\*outputs)
        Declare specific outputs as carrying no gradient (e.g.
        integer indices, masks).
    set_materialize_grads(value)
        Reserved hook for controlling whether ``None`` upstream
        gradients are materialised as zero tensors before
        ``backward`` is invoked.

    Notes
    -----
    The context is what ties the forward and backward halves of a
    custom node together in the chain rule:

    .. math::

        \mathbf{y} = f(\mathbf{x}; \text{ctx}),
        \qquad
        \bar{\mathbf{x}} = g(\mathbf{ctx}, \bar{\mathbf{y}}),

    where :math:`\bar{\mathbf{y}} = \partial \mathcal{L} /
    \partial \mathbf{y}` is the upstream gradient and the same
    ``ctx`` object is passed to both halves so :math:`g` can read
    back whatever :math:`f` saved.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import Function
    >>> class Square(Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         ctx.shape = x.shape
    ...         return x * x
    ...     @staticmethod
    ...     def backward(ctx, grad_out):
    ...         (x,) = ctx.saved_tensors
    ...         return 2 * x * grad_out
    """

    def __init__(self) -> None:
        """Initialise an empty context with no saved tensors or extras."""
        self._saved_tensors: list[Tensor] = []
        # One entry per saved tensor — its impl, that impl's version once
        # ``forward`` returned, and where it came from for the error message
        # — written when the node is registered.  ``None`` before that, so a
        # ``forward`` reading its own ``saved_tensors`` is not checked.
        self._saved_record: list[_SavedEntry | None] | None = None
        self.needs_input_grad: tuple[bool, ...] = ()
        self._non_differentiable: list[Tensor] = []
        self._extra: dict[str, object] = {}

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Store tensors needed to compute the backward pass.

        Parameters
        ----------
        *tensors : Tensor
            Activations / inputs the ``backward`` implementation will
            require. They are retrieved later via the :attr:`saved_tensors`
            property as a tuple in the same order.

        Notes
        -----
        Each call replaces any tensors previously saved on this context.

        A saved tensor must not be written in place between the end of
        ``forward`` and the ``backward`` that reads it: reading
        :attr:`saved_tensors` then raises ``VersionMismatch``, as a built-in
        op's backward does, instead of handing ``backward`` the new values.
        Writes inside ``forward`` itself are fine — the version is taken
        once ``forward`` has returned.
        """
        self._saved_tensors = list(tensors)

    def _record_saved_versions(
        self, name: str, inputs: tuple[object, ...], returned: tuple[object, ...]
    ) -> None:
        """Note each saved tensor's impl and version now that ``forward`` is done.

        Parameters
        ----------
        name : str
            The ``Function`` subclass's name, for the error message.
        inputs : tuple of object
            ``forward``'s positional arguments.
        returned : tuple of object
            What ``forward`` returned, as a tuple.

        Notes
        -----
        Taken after ``forward`` — the reference framework saves at the same
        point — so a ``forward`` that writes a tensor in place and then
        returns it is not refused for its own write.
        """
        input_impls = [_impl_of(a) for a in inputs]
        output_impls = [_impl_of(o) for o in returned]
        record: list[_SavedEntry | None] = []
        for i, t in enumerate(self._saved_tensors):
            impl = _impl_of(t)
            if impl is None:
                record.append(None)
                continue
            # Named the way the engine names a node's input — "mul input 1" —
            # by the position the caller knows it under.
            k = _position(impl, input_impls)
            j = _position(impl, output_impls)
            if k is not None:
                where = f"{name} input {k}"
            elif j is not None:
                where = f"{name} output {j}"
            else:
                where = f"{name} saved tensor {i}"
            record.append((impl, impl.version, where))
        self._saved_record = record

    @property
    def saved_tensors(self) -> tuple[Tensor, ...]:
        """Read back the tensors saved during ``forward``.

        Returns
        -------
        tuple of Tensor
            The tensors stored by :meth:`save_for_backward`, wrapped
            back into Python ``Tensor`` instances if the engine stored
            raw ``TensorImpl`` handles.

        Raises
        ------
        VersionMismatch
            A saved tensor was written in place after ``forward``
            returned — its values are no longer the ones ``forward``
            saw, so a gradient computed from them would be wrong.
            Suppressed inside
            ``lucid.autograd.graph.allow_mutation_on_saved_tensors``.
        """
        record = self._saved_record
        check = record is not None and not _C_engine.is_mutation_on_saved_allowed()
        result: list[Tensor] = []
        for i, t in enumerate(self._saved_tensors):
            entry = record[i] if record is not None else None
            if entry is None:
                result.append(_wrap(t) if isinstance(t, _C_engine.TensorImpl) else t)
                continue
            impl, version, where = entry
            if check and impl.version != version:
                # Worded as the engine's own ``check_version_match`` words it.
                raise _C_engine.VersionMismatch(
                    f"VersionMismatch ({where}): saved version {version} but "
                    f"tensor is now at version {impl.version} (in-place "
                    "mutation between forward and backward?)"
                )
            # What was saved is the impl, as for a built-in op: a write that
            # rebinds the Python tensor to a new impl (``t[i] = v`` on a
            # tensor that needs no grad) leaves the saved values where they
            # were, and ``backward`` reads those, not the new ones.
            if isinstance(t, Tensor) and t._impl is impl:
                result.append(t)
            else:
                result.append(_wrap(impl))
        return tuple(result)

    def mark_non_differentiable(self, *tensors: Tensor) -> None:
        """Declare that the given output tensors carry no gradient.

        Parameters
        ----------
        *tensors : Tensor
            Outputs of :meth:`Function.forward` for which autograd should
            not propagate gradients (e.g., integer indices, masks).

        Notes
        -----
        :meth:`Function.apply` hands a marked output back detached:
        ``requires_grad=False`` and no ``grad_fn``, and a compiled step
        sends no gradient through it either.  ``backward`` is still given a
        gradient for it — zeros — so its arguments keep one slot per
        output.  Marking one of ``forward``'s inputs, returned as-is,
        leaves the caller's tensor alone: what comes back is a new tensor
        over the same storage.

        Examples
        --------
        >>> import lucid
        >>> from lucid.autograd import Function
        >>> class ScaleAndMask(Function):
        ...     @staticmethod
        ...     def forward(ctx, x, mask):
        ...         ctx.mark_non_differentiable(mask)
        ...         return x * 2, mask
        ...     @staticmethod
        ...     def backward(ctx, grad, grad_mask):
        ...         return grad * 2, None
        >>> x = lucid.ones(2, requires_grad=True)
        >>> mask = lucid.ones(2, requires_grad=True)
        >>> y, out = ScaleAndMask.apply(x, mask)
        >>> out.requires_grad, mask.requires_grad
        (False, True)
        """
        self._non_differentiable = list(tensors)

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Route user-defined attributes onto the ``_extra`` dict.

        Reserved private names and ``needs_input_grad`` use normal
        instance storage so they remain accessible through descriptors;
        everything else falls back to the extras bag, keeping the
        context permissive while preserving the public schema.
        """
        if name.startswith("_") or name in ("needs_input_grad",):
            object.__setattr__(self, name, value)
        else:
            try:
                object.__setattr__(self, name, value)
            except AttributeError:
                self._extra[name] = value

    def __getattr__(self, name: str) -> object:
        """Look up overflow attributes that were stored in ``_extra``.

        Called only when normal attribute resolution fails, so it
        complements :meth:`__setattr__` without shadowing class members.
        """
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        raise AttributeError(f"FunctionCtx has no attribute '{name}'")


def _owned_outputs(
    entries: list[object], ctx: FunctionCtx, inputs: list[Tensor]
) -> tuple[list[object], list[Tensor], list[bool]]:
    """``forward``'s results, with every marked one a tensor this call owns.

    A marked output is detached, which rebinds the tensor it is applied
    to.  A tensor ``forward`` made is its own, and so is a view it took of
    an input; one of the caller's inputs returned as-is is not, and is
    replaced by the detached alias :meth:`Tensor.detach` hands out — a new
    tensor over the same storage, as the reference hands back — so the
    caller's tensor keeps its ``requires_grad``.

    An unmarked input returned as-is is still handed back itself, with the
    node registered on it.  Neither alias the engine can make keeps both
    gradients right: a compiled step traces the detached one as a gradient
    sink, and an in-place write to the input re-derives a view of it from
    the write, dropping the custom ``backward``.  The reference's view needs
    a mark the engine does not have yet (LCD-558).

    Parameters
    ----------
    entries : list of object
        What ``forward`` returned, one entry per result.
    ctx : FunctionCtx
        The context ``forward`` filled in, with its marks.
    inputs : list of Tensor
        The caller's tensors: every tensor passed to ``apply``,
        positionally or by keyword.

    Returns
    -------
    tuple
        The entries with each marked input swapped for its alias, their
        tensors in order, and whether each tensor was marked
        non-differentiable.
    """
    input_impls: list[_C_engine.TensorImpl | None] = [t._impl for t in inputs]
    marked_impls: list[_C_engine.TensorImpl | None] = [
        t._impl for t in ctx._non_differentiable
    ]
    owned: list[object] = []
    tensors: list[Tensor] = []
    marked: list[bool] = []
    for entry in entries:
        if isinstance(entry, Tensor):
            # Marks name what ``forward`` returned, so they are read before
            # an input is swapped for its alias.
            is_marked = _position(entry._impl, marked_impls) is not None
            if is_marked and _position(entry._impl, input_impls) is not None:
                entry = _wrap(entry._impl.data_alias())
            marked.append(is_marked)
            tensors.append(entry)
        owned.append(entry)
    return owned, tensors, marked


def _wrap_outputs(
    output: object,
    klass: type,
    ctx: FunctionCtx,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracked: bool,
) -> object:
    """What ``apply`` hands back for what ``forward`` returned.

    Detaching a marked output rebinds the tensor it is applied to, so it is
    applied only to tensors this call owns (:func:`_owned_outputs`): a
    marked input keeps its ``requires_grad``.

    Parameters
    ----------
    output : object
        What ``forward`` returned: a tensor, or a tuple that may hold
        non-tensor entries alongside the tensors.
    klass : type
        The ``Function`` subclass, whose ``backward`` the node calls.
    ctx : FunctionCtx
        The context ``forward`` filled in.
    args : tuple of object
        ``forward``'s positional arguments, the ones gradients go to.
    kwargs : dict of str to object
        ``forward``'s keyword arguments.  They receive no gradient, but a
        tensor among them is the caller's as much as a positional one.
    tracked : bool
        Whether this call is recorded — grad mode is on and an input needs
        a gradient.

    Returns
    -------
    object
        ``output`` with each tensor entry settled: on the node when it is
        differentiable and the call is tracked, detached when it was
        marked non-differentiable.
    """
    is_tuple = isinstance(output, tuple)
    tensor_inputs = [a for a in args if isinstance(a, Tensor)]
    callers = tensor_inputs + [v for v in kwargs.values() if isinstance(v, Tensor)]
    entries, tensors, marked = _owned_outputs(
        list(output) if isinstance(output, tuple) else [output], ctx, callers
    )
    if tracked and not all(marked):
        # Every tensor output is on the node, a marked one too: ``backward``
        # is handed one gradient per *tensor* output, zeros for a marked one.
        on_node = tuple(tensors) if is_tuple else tensors[0]
        # Which positional arguments were tensors, so ``backward`` may
        # answer with one gradient per argument (``None`` for the rest).
        is_tensor_arg = [isinstance(a, Tensor) for a in args]
        _register(on_node, klass, ctx, tensor_inputs, is_tensor_arg)  # type: ignore[arg-type]
    for t, is_marked in zip(tensors, marked, strict=True):
        # Whether or not it needs a gradient here: a compiled step traces the
        # model under ``no_grad``, and only the detach tells it to stop.
        if is_marked:
            t.detach_()
    return tuple(entries) if is_tuple else entries[0]


def _make_apply() -> classmethod:  # type: ignore[type-arg]
    """Build the per-subclass ``apply`` classmethod injected by ``FunctionMeta``."""

    def apply(
        klass: type, *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor, ...]:
        """Run ``forward`` and register the autograd node when needed."""
        ctx = FunctionCtx()
        ctx.needs_input_grad = tuple(
            isinstance(a, Tensor) and a.requires_grad for a in args
        )

        output = klass.forward(ctx, *args, **kwargs)  # type: ignore[attr-defined]
        if (tracer := _C_engine.compile.current_tracer()) is not None:
            # Imported here: ``lucid.compile`` imports this package.
            from lucid.compile._core.custom_backward import note_applied

            note_applied(tracer, klass, args, output)

        tracked = _C_engine.grad_enabled() and any(ctx.needs_input_grad)
        output = _wrap_outputs(output, klass, ctx, args, kwargs, tracked)
        if tracked:
            # The node exists now, so ``backward`` will read the saved
            # tensors: note their versions, as a built-in op does when it
            # saves its inputs, so a write in between is refused there.
            returned = output if isinstance(output, tuple) else (output,)
            ctx._record_saved_versions(klass.__name__, args, returned)

        return cast(Tensor | tuple[Tensor, ...], output)

    return classmethod(apply)  # type: ignore[arg-type]


class FunctionMeta(type):
    """Metaclass that wires a fresh :meth:`apply` onto each ``Function`` subclass.

    The base ``Function`` class is left untouched (so ``Function.apply`` keeps
    its explanatory stub); every concrete subclass receives a closure that
    instantiates a :class:`FunctionCtx`, calls the subclass's ``forward``,
    and — when grad is enabled and any input requests it — registers the
    autograd node.
    """

    def __init__(
        cls, name: str, bases: tuple[type, ...], dct: dict[str, object]
    ) -> None:
        """Install the dispatching :meth:`apply` on the freshly built subclass."""
        super().__init__(name, bases, dct)
        if name != "Function":
            cls.apply = _make_apply()


class Function(metaclass=FunctionMeta):
    r"""Base class for custom differentiable operations.

    Subclass to define an operation whose forward and backward
    passes Lucid's autograd cannot deduce automatically — for
    example, an operation that wraps an external library, an op
    with a custom backward formula for numerical stability, or a
    piecewise-defined function whose gradient differs from naïve
    autograd.

    Define :py:meth:`forward` and :py:meth:`backward` as
    ``@staticmethod`` on the subclass, then invoke the op via
    :py:meth:`apply` (NOT by calling ``forward`` directly — that
    would skip autograd registration).

    Parameters
    ----------
    None
        ``Function`` itself is never instantiated. Subclasses
        are stateless — their behaviour is defined entirely by
        the ``forward`` / ``backward`` static methods, and ops
        are invoked through the :py:meth:`apply` classmethod
        rather than ``__init__``.

    Attributes
    ----------
    forward : staticmethod
        ``forward(ctx, *args, **kwargs) -> Tensor or tuple[Tensor, ...]``.
        Computes the primal value; stores anything ``backward`` will
        need on ``ctx``.
    backward : staticmethod
        ``backward(ctx, *grad_outputs) -> tuple of Tensors (or None per
        non-tensor input)``. Returns the cotangents matching the
        positional inputs of ``forward``.
    apply : classmethod
        Bind ``forward`` to the autograd graph, returning the forward
        output and registering ``backward`` as the gradient closure.

    Notes
    -----
    The :py:class:`FunctionCtx` passed to ``forward`` carries the
    bookkeeping autograd needs to call ``backward`` later — saved
    tensors plus per-input non-differentiable flags.

    An input that ``forward`` returns as-is and marks non-differentiable —
    passed positionally or by keyword — comes back as a new, detached
    tensor over the same storage, and the caller's tensor keeps its
    ``requires_grad``.  An unmarked one comes back as the same tensor, now
    the node's output: where the reference framework hands back a view of
    it, the caller's tensor here is no longer a leaf after the call.

    Each :py:class:`Function` defines a node in the computation graph

    .. math::

        \mathbf{y} = f(\mathbf{x}_1, \ldots, \mathbf{x}_n; \theta)

    and supplies the chain-rule contribution

    .. math::

        \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i}
        = \sum_j \frac{\partial \mathcal{L}}{\partial y_j}
          \cdot \frac{\partial y_j}{\partial x_{i}}

    for each input :math:`\mathbf{x}_i`. The ``backward`` method
    implements precisely this Jacobian-vector product.

    Examples
    --------
    Custom ReLU with an explicit backward:

    >>> import lucid
    >>> from lucid.autograd import Function
    >>> class MyReLU(Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         return x.clamp(min=0.0)
    ...     @staticmethod
    ...     def backward(ctx, grad_out):
    ...         (x,) = ctx.saved_tensors
    ...         return grad_out * (x > 0).float()
    >>> y = MyReLU.apply(lucid.tensor([-1.0, 2.0], requires_grad=True))
    >>> y.sum().backward()
    """

    # The base signature is ``(ctx, *Any, **Any) -> Any`` on purpose: every
    # subclass declares its own concrete parameters and return type, and only
    # this shape is override-compatible with all of them (LSP), so subclasses
    # need no ``ignore[override]``.  The runtime contract is documented below
    # and enforced by ``apply`` / the backward node, not by this annotation.
    @staticmethod
    def forward(ctx: FunctionCtx, *args: Any, **kwargs: Any) -> Any:
        """Compute the forward result of the custom op.

        Subclasses override this static method. Save anything needed
        during backward on ``ctx`` (via ``ctx.save_for_backward(...)``
        or by setting attributes like ``ctx.shape = x.shape``); do not
        capture tensors via closure.

        Parameters
        ----------
        ctx : FunctionCtx
            Per-call context; will be passed unchanged to :meth:`backward`.
        *args : Tensor
            Positional inputs to the custom op. Non-Tensor positional
            arguments are allowed but receive no gradients.
        **kwargs
            Keyword inputs forwarded unchanged from :meth:`apply`; they
            receive no gradients.

        Returns
        -------
        Tensor or tuple of Tensor
            One or more output tensors.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: Any) -> Any:
        r"""Compute gradients of the loss w.r.t. each input of :meth:`forward`.

        Subclasses override this static method. The returned tuple must
        contain one gradient per **positional argument** of ``forward``,
        in the same order; non-Tensor arguments receive ``None``.

        Parameters
        ----------
        ctx : FunctionCtx
            The same context that was populated during ``forward``.
        *grad_outputs : Tensor
            Upstream gradients :math:`\partial L / \partial y_i`, one per
            output that ``forward`` returned.

        Returns
        -------
        Tensor or tuple of Tensor
            Downstream gradients matching the positional inputs of
            ``forward``. Use ``None`` for inputs that have no gradient.
            One gradient per *tensor* input — leaving out the non-tensor
            arguments — is accepted as well, and trailing ``None``
            entries past the last argument are ignored.

        Raises
        ------
        TypeError
            When a gradient is neither a tensor nor ``None``.
        RuntimeError
            When the number of gradients fits neither form, when a
            non-tensor argument is given a gradient other than ``None``,
            or when a gradient does not fit its input: a shape the input
            does not broadcast to, or a device other than the input's.

        Notes
        -----
        Each gradient is checked against the input it is for, once
        ``backward`` returns.  A shape the input broadcasts to is summed
        back down to the input's shape, another dtype is cast to the
        input's (a complex gradient for a real input keeps its real part),
        and a 0-d gradient on another device is moved to the input's.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: object, **kwargs: object) -> object:
        """Run the function — autograd-tracked when ``forward`` requests grad.

        ``FunctionMeta`` rewrites this on every concrete subclass to dispatch
        through ``forward``/``backward``; the base implementation here exists
        only so ``Function.apply`` is a real attribute (matters for tooling
        and ``hasattr`` checks).  Calling it on the base class is meaningless
        and raises.
        """
        raise NotImplementedError(
            "Function.apply must be called on a concrete subclass that "
            "implements forward / backward."
        )
