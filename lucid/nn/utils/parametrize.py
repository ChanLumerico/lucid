"""Parametrization API — register a transformation that re-derives a
parameter on every forward pass.

This is the modern replacement for the legacy ``weight_norm`` /
``spectral_norm`` machinery.  ``register_parametrization`` swaps a leaf
``Parameter`` for a derived tensor computed by an arbitrary callable; the
callable's own parameters become the trained quantities and the original
weight is exposed read-only as ``module.parametrizations[name].original``.

Mirrors the public surface of ``reference framework.nn.utils.parametrize``.  The
implementation here covers the common case used in production code:
single-parametrization registration, ``is_parametrized`` introspection, and
``remove_parametrizations`` to fold the derived tensor back into a plain
leaf parameter.
"""

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter

# Sentinel attribute on a parametrized module.  Maps original-parameter
# name → ``ParametrizationContainer``; the container in turn holds the
# transformation module and the underlying ``original`` parameter.
_PARAM_HOOK_ATTR: str = "parametrizations"


class ParametrizationContainer(Module):
    """Holds one ``Parametrization`` (the transformation) plus the
    untransformed weight tensor as ``original``.

    The container is attached to ``module.parametrizations[name]`` so users
    can introspect / mutate the underlying parameter through the standard
    module-tree mechanisms.
    """

    def __init__(self, parametrization: Module, original: Parameter) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        super().__init__()
        self.parametrization: Module = parametrization
        self.original: Parameter = original

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        # Apply the transformation to the cached ``original`` weight.
        """Apply the layer / parametrization to the input."""
        return self.parametrization(self.original)  # type: ignore[return-value]


def register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False,
) -> Module:
    r"""Install a parametrization on a parameter of ``module``.

    After registration, ``module.<tensor_name>`` is no longer a leaf
    parameter — every read invokes ``parametrization(original)`` where
    ``original`` is the cached pre-transformation weight.  The original
    becomes the actual trainable Parameter, exposed at
    ``module.parametrizations[tensor_name].original``.  This is the
    modern, general-purpose alternative to :func:`weight_norm` /
    :func:`spectral_norm`: any differentiable transformation of a weight
    can be installed.

    Parameters
    ----------
    module : Module
        Host module whose parameter is to be parametrised.  Mutated in
        place.
    tensor_name : str
        Attribute name of the parameter to wrap.
    parametrization : Module
        A module whose ``forward(W) -> W'`` defines the transformation.
        Its own parameters (if any) become trainable; ``W'`` is what
        ``module.<tensor_name>`` returns at access time.
    unsafe : bool, keyword-only, optional
        If ``True``, skip the sanity check that
        ``parametrization(original)`` produces the same shape as the
        original.  Required for shape-changing parametrisations (e.g.
        Householder factorisations).

    Returns
    -------
    Module
        The same ``module``, now with the parametrisation attached.

    Raises
    ------
    AttributeError
        If ``tensor_name`` is not present on the module.
    RuntimeError
        If a parametrisation is already registered on ``tensor_name``
        (chaining is not yet supported) or, when ``unsafe`` is ``False``,
        the produced tensor shape differs from the original.

    Notes
    -----
    Mathematically, calling ``register_parametrization(m, 'weight', f)``
    re-expresses the trained quantity as

    .. math::

        \mathbf{W} \;=\; f(\boldsymbol\theta), \qquad
        \boldsymbol\theta \in \mathcal{M},

    where :math:`\boldsymbol\theta` is the new ``original`` parameter
    (now living on a possibly constrained manifold :math:`\mathcal{M}`)
    and :math:`f` is the parametrisation module.  Gradients flow
    through :math:`f` automatically during backprop.

    Examples
    --------
    >>> from lucid.nn.utils.parametrize import register_parametrization
    >>> class Symmetric(nn.Module):
    ...     def forward(self, X):
    ...         return 0.5 * (X + X.mT)
    >>> register_parametrization(layer, "weight", Symmetric())
    """
    if not hasattr(module, tensor_name):
        raise AttributeError(f"module has no parameter '{tensor_name}'")
    weight: Parameter = getattr(module, tensor_name)
    if not isinstance(weight, Parameter):
        raise TypeError(
            f"'{tensor_name}' must be a Parameter to parametrise, got "
            f"{type(weight).__name__}"
        )

    container_dict: dict[str, ParametrizationContainer] = getattr(
        module, _PARAM_HOOK_ATTR, {}
    )
    if tensor_name in container_dict:
        raise RuntimeError(
            f"'{tensor_name}' is already parametrised; chaining is not yet supported"
        )

    container: ParametrizationContainer = ParametrizationContainer(
        parametrization, Parameter(weight.detach())
    )
    container_dict[tensor_name] = container
    object.__setattr__(module, _PARAM_HOOK_ATTR, container_dict)

    # Drop the leaf Parameter so the property below shadows it.  The
    # container's ``original`` is the new trainable Parameter.
    del module._parameters[tensor_name]

    def _pre_hook(mod: Module, inputs: object) -> None:  # noqa: ANN401
        cont: ParametrizationContainer = getattr(mod, _PARAM_HOOK_ATTR)[tensor_name]
        object.__setattr__(mod, tensor_name, cont())

    handle = module.register_forward_pre_hook(_pre_hook)  # type: ignore[arg-type]
    # Stash the hook handle on the container so ``remove_parametrizations``
    # can detach it later.
    container._hook_handle = handle

    # Materialise the derived tensor immediately so attribute access works
    # before any forward call.
    object.__setattr__(module, tensor_name, container())

    if not unsafe:
        produced: Tensor = container()  # type: ignore[assignment]
        if tuple(produced.shape) != tuple(weight.shape):
            raise RuntimeError(
                f"parametrisation produced shape {tuple(produced.shape)} but "
                f"original was {tuple(weight.shape)}; pass unsafe=True if intentional"
            )
    return module


def is_parametrized(module: Module, tensor_name: str | None = None) -> bool:
    r"""Predicate: does ``module`` carry an active parametrisation?

    Lightweight introspection — checks for the presence of the
    ``parametrizations`` container without invoking any forward hook.
    Use this in user code that needs to behave differently for raw vs
    reparametrised layers (e.g. checkpoint serialisation, weight
    initialisation utilities).

    Parameters
    ----------
    module : Module
        Module to inspect.
    tensor_name : str, optional
        If given, narrow the check to a specific parameter name.  When
        ``None`` (the default), return ``True`` if *any* parameter on
        the module is parametrised.

    Returns
    -------
    bool
        ``True`` if a parametrisation exists matching the query,
        ``False`` otherwise.

    Notes
    -----
    Cheap to call — runs in :math:`O(1)`; no tensor work is performed.

    Examples
    --------
    >>> register_parametrization(m, "weight", Symmetric())
    >>> is_parametrized(m)
    True
    >>> is_parametrized(m, "bias")
    False
    """
    container_dict: dict[str, object] = getattr(module, _PARAM_HOOK_ATTR, {})
    if not container_dict:
        return False
    if tensor_name is None:
        return True
    return tensor_name in container_dict


def remove_parametrizations(
    module: Module,
    tensor_name: str,
    leave_parametrized: bool = True,
) -> Module:
    r"""Reverse :func:`register_parametrization` and restore a leaf parameter.

    Detaches the forward pre-hook and reinstalls ``module.<tensor_name>``
    as a plain :class:`~lucid.nn.parameter.Parameter`.  Choose whether to
    keep the most recent *transformed* value or roll back to the raw
    pre-parametrisation weight via ``leave_parametrized``.

    Parameters
    ----------
    module : Module
        Module previously passed through :func:`register_parametrization`.
    tensor_name : str
        Attribute name to un-parametrise.  Must match the name used at
        registration.
    leave_parametrized : bool, optional
        If ``True`` (the default), the new leaf parameter is set to
        :math:`f(\boldsymbol\theta)` — the most recent output of the
        parametrisation.  If ``False``, the underlying
        :math:`\boldsymbol\theta` (the ``original`` Parameter) is
        restored as-is.

    Returns
    -------
    Module
        The same ``module`` with the parametrisation removed.

    Raises
    ------
    ValueError
        If no parametrisation is registered on ``tensor_name``.

    Notes
    -----
    The restored Parameter is detached from the autograd graph — any
    gradient history accumulated through the parametrisation is
    discarded.  If multiple parametrisations are present, only the one
    matching ``tensor_name`` is removed; the others remain attached.

    Examples
    --------
    >>> remove_parametrizations(layer, "weight")
    >>> is_parametrized(layer, "weight")
    False
    """
    container_dict: dict[str, ParametrizationContainer] = getattr(
        module, _PARAM_HOOK_ATTR, {}
    )
    if tensor_name not in container_dict:
        raise ValueError(f"no parametrisation registered on '{tensor_name}'")
    container: ParametrizationContainer = container_dict.pop(tensor_name)
    handle = container._hook_handle
    handle.remove()  # type: ignore[union-attr]

    final_value: Tensor = (
        container() if leave_parametrized else container.original.detach()  # type: ignore[assignment]
    )

    # Drop the cached non-leaf attribute so ``register_parameter`` succeeds.
    if tensor_name in module.__dict__:
        del module.__dict__[tensor_name]
    module.register_parameter(tensor_name, Parameter(final_value.detach()))

    if not container_dict:
        try:
            object.__delattr__(module, _PARAM_HOOK_ATTR)
        except AttributeError:
            pass
    return module
