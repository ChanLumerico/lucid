"""
autocast context manager for automatic mixed precision.
"""

import functools
import threading
from typing import Callable
from lucid._C import engine as _C_engine
from lucid._dtype import dtype, float16, to_engine_dtype

#: The device types an autocast scope can name — the same two ``lucid.device``
#: accepts.
_DEVICE_TYPES = frozenset({"metal", "cpu"})


class autocast:
    """Enable automatic mixed-precision computation inside a context.

    Operations that opt into AMP (matmul, conv, attention, …) cast their
    inputs to the target lower-precision dtype on entry, reducing memory
    pressure and — on Metal — improving throughput by activating the
    M-series GPU's half-precision tensor pipelines.  Cast points are
    inserted *only* at op boundaries; activations themselves are stored
    in the lower-precision dtype but accumulator and reduction state stay
    in float32 to preserve numerical fidelity (the standard "mixed" recipe).

    Leaving the scope restores exactly the state it was entered from —
    autocast off, or the enclosing scope's dtype — whether it is left
    normally or by an exception, and however long the ``autocast`` object
    itself lives on afterwards.  One object can be entered again after it
    exits, entered while it is already active (each entry is undone by its
    own exit), and used as a decorator.  ``enabled=False`` leaves the
    current state untouched: it does not switch an enclosing scope off.

    The engine keeps one autocast state per thread rather than one per
    device, so a scope currently applies to eligible ops on *both*
    devices whichever ``device_type`` it names (CHA-70).

    Parameters
    ----------
    device_type : str, optional
        Device the autocast scope applies to.  ``"metal"`` (GPU stream,
        default) or ``"cpu"`` (Accelerate stream); anything else raises
        :class:`ValueError`.
    dtype : lucid.dtype, optional
        Lower-precision dtype that supported ops cast inputs to inside
        the scope.  Default :data:`lucid.float16`.  On the CPU, float16
        runs in float32 — Accelerate has no half-precision kernels — so a
        CPU scope with the default dtype changes nothing; pass
        ``dtype=lucid.bfloat16`` for reduced precision there, as the
        reference framework's CPU autocast does by default.  Not every CPU
        op has a bfloat16 kernel yet.
    enabled : bool, optional
        When ``False`` the context is a no-op — useful for ablation /
        A/B comparing AMP on vs off without changing call sites.
        Default ``True``.

    Examples
    --------
    Context-manager form (typical training loop):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> model = nn.Linear(4, 2).to("metal")
    >>> x = lucid.randn(8, 4, device="metal")
    >>> with lucid.amp.autocast():
    ...     output = model(x)            # ops cast to float16 inside
    >>> output.dtype
    lucid.float16

    Decorator form (every call wraps itself in a fresh scope):

    >>> @lucid.amp.autocast()
    ... def predict(x):
    ...     return model(x)
    >>> predict(x).dtype
    lucid.float16

    Nested scopes restore the outer dtype on exit:

    >>> a = lucid.randn(2, 2, device="metal")
    >>> with lucid.amp.autocast(dtype=lucid.float16):
    ...     print((a @ a).dtype)  # fp16 here
    ...     with lucid.amp.autocast(dtype=lucid.bfloat16):
    ...         print((a @ a).dtype)  # bf16 here
    ...     print((a @ a).dtype)  # back to fp16 — prior guard reinstalled
    lucid.float16
    lucid.bfloat16
    lucid.float16

    See Also
    --------
    lucid.amp.GradScaler : loss-scaling counterpart for fp16 training.
    """

    def __init__(
        self,
        device_type: str = "metal",
        dtype: dtype = float16,
        enabled: bool = True,
    ) -> None:
        """Configure the autocast context.

        Parameters
        ----------
        device_type : str, default='metal'
            Device the autocast scope applies to. Apple Silicon supports
            ``'metal'`` (GPU stream) and ``'cpu'`` (Accelerate stream).
        dtype : lucid.dtype, default=float16
            Lower-precision dtype that supported ops cast inputs to inside
            the scope.
        enabled : bool, default=True
            When ``False`` the context is a no-op (useful for ablation).

        Raises
        ------
        ValueError
            If ``device_type`` is neither ``'metal'`` nor ``'cpu'``.
        """
        if device_type not in _DEVICE_TYPES:
            raise ValueError(
                f"autocast: unknown device_type {device_type!r}. Use 'metal' or 'cpu'."
            )
        self._device_type = device_type
        self._dtype = dtype
        self._enabled = enabled
        # One stack of live engine guards per thread: the AMP state the
        # guards change is thread-local, so a guard must be released on the
        # thread that made it.  A stack (not a single slot) lets the same
        # object be entered again while it is already active.
        self._local = threading.local()

    def _guards(self) -> list[_C_engine.AutocastGuard]:
        """Return this thread's stack of guards opened by this object."""
        stack: list[_C_engine.AutocastGuard] | None = getattr(
            self._local, "stack", None
        )
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def __enter__(self) -> autocast:
        """Activate the autocast scope and return ``self``.

        Constructing the engine ``AutocastGuard`` switches AMP on with the
        configured dtype and records the state it replaced; the guard is
        kept until :meth:`__exit__`.
        """
        if not self._enabled:
            return self
        self._guards().append(_C_engine.AutocastGuard(to_engine_dtype(self._dtype)))
        return self

    def __exit__(self, *args: object) -> None:
        """Restore the AMP state that was active when the scope was entered.

        Releases the guard :meth:`__enter__` made.  Its destructor runs at
        once and puts back the state it recorded, so the scope ends here
        rather than whenever the ``autocast`` object is collected.
        Exceptions propagate.
        """
        if not self._enabled:
            return
        guard = self._guards().pop()
        del guard  # the last reference — ~AutocastGuard restores the prior state

    def __call__[F: Callable[..., object]](self, fn: F) -> F:
        """Use as a function decorator.

        Every call of the decorated function runs inside this scope — the
        same ``device_type``, ``dtype`` and ``enabled`` — and leaves it on
        return or on an exception.

        Parameters
        ----------
        fn : callable
            The function to wrap.

        Returns
        -------
        callable
            ``fn`` wrapped so that each call enters and exits the scope.
        """

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            """Invoke ``fn`` inside this autocast scope."""
            with self:
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    @staticmethod
    def is_autocast_enabled() -> bool:
        """Return True if autocast is currently active."""
        return _C_engine.amp_is_active()

    @staticmethod
    def get_autocast_dtype() -> dtype | None:
        """Return the currently active autocast dtype, or None."""
        from lucid._dtype import _ENGINE_TO_DTYPE

        eng = _C_engine.amp_active_dtype()
        if eng is None:
            return None
        return _ENGINE_TO_DTYPE.get(eng)
