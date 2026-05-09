"""Hook registration helpers for Module."""

from collections import OrderedDict
from typing import Callable

_GLOBAL_FORWARD_PRE_HOOKS: OrderedDict[int, tuple[Callable[..., object], bool]] = (
    OrderedDict()
)
_GLOBAL_FORWARD_HOOKS: OrderedDict[int, tuple[Callable[..., object], bool, bool]] = (
    OrderedDict()
)
_GLOBAL_BACKWARD_PRE_HOOKS: OrderedDict[int, Callable[..., object]] = OrderedDict()
_GLOBAL_BACKWARD_HOOKS: OrderedDict[int, Callable[..., object]] = OrderedDict()
# load_state_dict pre-hook signature:
#   hook(module, state_dict, prefix, local_metadata, strict, missing_keys,
#        unexpected_keys, error_msgs) -> None
_GLOBAL_LOAD_STATE_DICT_PRE_HOOKS: OrderedDict[int, Callable[..., object]] = (
    OrderedDict()
)
# load_state_dict post-hook signature:
#   hook(module, incompatible_keys) -> None
_GLOBAL_LOAD_STATE_DICT_POST_HOOKS: OrderedDict[int, Callable[..., object]] = (
    OrderedDict()
)

_HOOK_ID = 0


def _next_hook_id() -> int:
    global _HOOK_ID
    _HOOK_ID += 1
    return _HOOK_ID


class RemovableHandle:
    """Handle returned by register_*_hook(); call .remove() to deregister."""

    def __init__(
        self,
        hooks: dict[int, object],
        key: int,
        extra_sets: tuple[set[int], ...] = (),
    ) -> None:
        self._hooks = hooks
        self._key = key
        self._extra_sets = extra_sets

    def remove(self) -> None:
        """Remove this hook from the module."""
        self._hooks.pop(self._key, None)
        for extra_set in self._extra_sets:
            extra_set.discard(self._key)

    def __enter__(self) -> RemovableHandle:
        return self

    def __exit__(self, *args: object) -> None:
        self.remove()


def register_module_forward_pre_hook(
    hook: Callable[..., object],
    *,
    with_kwargs: bool = False,
) -> RemovableHandle:
    """Register a global hook called before every Module forward call."""
    key = _next_hook_id()
    _GLOBAL_FORWARD_PRE_HOOKS[key] = (hook, with_kwargs)
    return RemovableHandle(_GLOBAL_FORWARD_PRE_HOOKS, key)


def register_module_forward_hook(
    hook: Callable[..., object],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    """Register a global hook called after every Module forward call."""
    key = _next_hook_id()
    _GLOBAL_FORWARD_HOOKS[key] = (hook, with_kwargs, always_call)
    return RemovableHandle(_GLOBAL_FORWARD_HOOKS, key)


def register_module_full_backward_pre_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    """Register a global hook called before module backward hooks."""
    key = _next_hook_id()
    _GLOBAL_BACKWARD_PRE_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_BACKWARD_PRE_HOOKS, key)


def register_module_full_backward_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    """Register a global hook called after module backward pre-hooks."""
    key = _next_hook_id()
    _GLOBAL_BACKWARD_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_BACKWARD_HOOKS, key)


def register_module_load_state_dict_pre_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    """Register a global pre-hook called for each Module during load_state_dict.

    Hook signature:
        hook(module, state_dict, prefix, local_metadata, strict,
             missing_keys, unexpected_keys, error_msgs) -> None

    The hook may mutate `state_dict`, `missing_keys`, `unexpected_keys`,
    and `error_msgs` in place to influence the load.
    """
    key = _next_hook_id()
    _GLOBAL_LOAD_STATE_DICT_PRE_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_LOAD_STATE_DICT_PRE_HOOKS, key)


def register_module_load_state_dict_post_hook(
    hook: Callable[..., object],
) -> RemovableHandle:
    """Register a global post-hook called after each Module's load_state_dict.

    Hook signature:
        hook(module, incompatible_keys) -> None
    """
    key = _next_hook_id()
    _GLOBAL_LOAD_STATE_DICT_POST_HOOKS[key] = hook
    return RemovableHandle(_GLOBAL_LOAD_STATE_DICT_POST_HOOKS, key)
