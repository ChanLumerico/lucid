import lucid
import lucid.models
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

from lucid._tensor.tensor import Tensor


# model = models.coatnet_0()
# models.summarize(
#     model,
#     input_shape=(1, 3, 224, 224),
#     truncate_from=None,
#     # test_backward=True,
# )

# model_arr = models.get_model_names()
# print(f"\nTotal Models: {len(model_arr)}")

coat = models.coatnet_0()

s0 = coat.s0
s1 = coat.s1
s2 = coat.s2
s3 = coat.s3
s4 = coat.s4

x = lucid.random.randn(1, 64, 112, 112, requires_grad=True)

y = s1(x)
print("y shape:", y.shape)

y.backward()
print("X_grad shape:", x.grad.shape)
