import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

from lucid._tensor.tensor import Tensor


# model = models.convnext_v2_atto()
# models.summarize(
#     model,
#     input_shape=(1, 3, 224, 224),
#     truncate_from=None,
#     test_backward=True,
# )

# model_arr = models.get_model_names()
# print(f"\nTotal Models: {len(model_arr)}")


batch_size = 2
seq_length = 5
embed_dim = 16
num_heads = 4

Q = lucid.random.randn(batch_size, seq_length, embed_dim, requires_grad=True)
K = lucid.random.randn(batch_size, seq_length, embed_dim, requires_grad=True)
V = lucid.random.randn(batch_size, seq_length, embed_dim, requires_grad=True)


mha = nn.MultiHeadAttention(embed_dim, num_heads, dropout=0.1)

output = mha(Q, K, V)
output.backward()

print(output.shape)
print(Q.grad.shape, K.grad.shape, V.grad.shape)
