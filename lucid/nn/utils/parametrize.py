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
        super().__init__()
        self.parametrization: Module = parametrization
        self.original: Parameter = original

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        # Apply the transformation to the cached ``original`` weight.
        return self.parametrization(self.original)  # type: ignore[return-value]


def register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False,
) -> Module:
    """Re-parametrise ``module.<tensor_name>`` through ``parametrization``.

    After the call, ``module.<tensor_name>`` is no longer a leaf — every
    access runs ``parametrization(original)`` where ``original`` is the
    cached pre-transformation weight.  The cache lives at
    ``module.parametrizations[tensor_name].original`` and is the actual
    trainable Parameter.

    Setting ``unsafe=True`` skips the post-registration sanity check that
    the transformation produces a tensor with the same shape — useful for
    transformations that change the shape on purpose.
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
    """Return True if ``module`` has any parametrisation (or a specific one)."""
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
    """Reverse a previous ``register_parametrization`` call.

    With ``leave_parametrized=True`` (the default) the most recent value
    of the derived tensor is materialised into a fresh leaf Parameter;
    otherwise the pre-transformation ``original`` weight is restored.
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
