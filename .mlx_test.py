import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

lucid.random.seed(42)


x = lucid.random.randn(2, 2, requires_grad=True, device="gpu")

y = F.softmax(x)
print(y)
