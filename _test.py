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
num_heads = 3
seq_len = 4
dim = 8

# Random query, key, value tensors
query = lucid.random.randn(batch_size, num_heads, seq_len, dim, requires_grad=True)
key = lucid.random.randn(batch_size, num_heads, seq_len, dim, requires_grad=True)
value = lucid.random.randn(batch_size, num_heads, seq_len, dim, requires_grad=True)

# Optional causal mask (lower triangular matrix)
is_causal = True

# Optional attention mask (this could be set to a mask of -inf values)
attn_mask = None  # Can be set to a custom mask (e.g., for padding)

# Dropout probability
dropout_p = 0.1

# Compute the attention output
output = F.scaled_dot_product_attention(
    query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
)
output.backward()

# Print the output
print("Attention output shape:", output.shape)
print(output)

print(key.grad.shape, key.grad.shape, value.grad.shape)
