import copy
from typing import Any, Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure


__all__ = ["RMSprop", "Rprop"]


class RMSprop(optim.Optimizer): ...


class Rprop(optim.Optimizer): ...
