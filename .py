import lucid
import lucid.nn as nn


a = lucid.empty(2, 2, requires_grad=True)
print(a.__dict__)

nn.init.uniform_(a, 0, 1)
print(a.__dict__)
