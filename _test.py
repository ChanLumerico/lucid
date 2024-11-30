import lucid
from lucid._tensor import Tensor
import lucid.nn.functional as F


input_ = lucid.ones((1, 1, 4, 4), requires_grad=True)
weight = lucid.ones((1, 1, 2, 2), requires_grad=True)
bias = lucid.zeros((1,), requires_grad=True)

out = F._conv.conv2d(input_, weight, bias, stride=1, padding=0)
out.backward(keep_grad=True)

print(input_.grad)
print(weight.grad)
print(bias.grad)
