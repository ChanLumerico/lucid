"""
lucid.nn — neural-network module system.

Re-exports `Parameter` / `Buffer` and the `Module` family
(Module, Sequential, ModuleList, ModuleDict, ParameterList,
ParameterDict) plus class decorators (`auto_repr`,
`set_state_dict_pass_attr`).

Sub-namespaces (init, functional, modules, utils, ...) are added in
later phases.
"""

from __future__ import annotations

from lucid.nn.parameter import Parameter, Buffer
from lucid.nn.module import (
    Module,
    Sequential, ModuleList, ModuleDict,
    ParameterList, ParameterDict,
    auto_repr, set_state_dict_pass_attr,
)
from lucid.nn import init
from lucid.nn import functional
from lucid.nn.modules import *


__all__ = [
    "Parameter", "Buffer",
    "Module",
    "Sequential", "ModuleList", "ModuleDict",
    "ParameterList", "ParameterDict",
    "auto_repr", "set_state_dict_pass_attr",
    "init", "functional",
]
