from typing import Iterable

import lucid
import lucid.nn as nn
import lucid.optim as optim

from lucid.types import _OptimClosure, _Scalar


__all__ = ["Adamax", "Adagrad", "Adadelta", "Adafactor"]


class Adamax(optim.Optimizer): ...


class Adagrad(optim.Optimizer): ...


class Adadelta(optim.Optimizer): ...


class Adafactor(optim.Optimizer): ...
