import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

from lucid._tensor.tensor import Tensor


model = models.transformer_big()
models.summarize(
    model,
    input_shape=[(1, 50), (1, 100)],
    truncate_from=None,
    test_backward=True,
)

model_arr = models.get_model_names()
print(f"\nTotal Models: {len(model_arr)}")
