import lucid
import lucid.models as models

import time


model = models.transformer_base()
model.to("cpu")

t0 = time.time_ns()

models.summarize(
    model,
    input_shape=[(1, 500), (1, 500)],
    truncate_from=25,
    test_backward=True,
    do_eval=False,
)

t1 = time.time_ns()

print(f"[{model.device.upper()}] Elapsed time: {(t1 - t0) / 1e9} sec")
