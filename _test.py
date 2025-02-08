import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.models as models

from lucid._tensor.tensor import Tensor


model = models.transformer_big()
# models.summarize(
#     model,
#     input_shape=(1, 3, 224, 224),
#     truncate_from=None,
#     test_backward=True,
# )
print(f"{model.parameter_size:,}")

model_arr = models.get_model_names()
print(f"\nTotal Models: {len(model_arr)}")
