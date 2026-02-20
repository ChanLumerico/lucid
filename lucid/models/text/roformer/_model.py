from dataclasses import dataclass
from typing import Callable

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.models.base import PreTrainedModelMixin
