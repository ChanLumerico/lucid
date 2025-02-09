import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

from lucid._tensor.tensor import Tensor


# model = models.transformer_big()
# models.summarize(
#     model,
#     input_shape=(1, 3, 224, 224),
#     truncate_from=None,
#     test_backward=True,
# )
# print(f"{model.parameter_size:,}")

# model_arr = models.get_model_names()
# print(f"\nTotal Models: {len(model_arr)}")


x = lucid.ones(6, 7, 8, requires_grad=True)
y = lucid.einops.rearrange(x, "a b (c d) -> (a b) c d", c=2, d=4)
z = lucid.einops.reduce(x, "a b c -> b a")
w = lucid.einops.repeat(x, "a b c -> a b c d", d=4)

y.backward()
z.backward()
w.backward()

print(y.shape)
print(z.shape)
print(w.shape)
