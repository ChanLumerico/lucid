import lucid
import lucid.nn.functional as F


input_ = lucid.ones((1, 1, 4, 4), requires_grad=True)
weight = lucid.ones((1, 1, 3, 3), requires_grad=True)
bias = lucid.zeros((1,), requires_grad=True)

out = F._conv._im2col_2d(input_, (3, 3), 1, 0)
out.backward()
