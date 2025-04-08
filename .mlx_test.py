import lucid
import lucid.models as models

import time


model = models.inception_v3()
model.to("gpu")

t0 = time.time_ns()

models.summarize(
    model,
    input_shape=(1, 3, 224, 224),
    truncate_from=100,
    test_backward=True,
    do_eval=False,
)

t1 = time.time_ns()

print(f"[{model.device.upper()}] Elapsed time: {(t1 - t0) / 1e6} ms")
