"""Metal's transposed-convolution input gradient, with output padding past the stride.

A transposed convolution's input gradient is the convolution it
transposes.  An ``output_padding`` of a stride or more — allowed while the
dilation is larger — lets that convolution fit one window past the input's
end, and Metal kept it: the extra row was read back as the input's shape,
so every row after the first came out shifted by one element.  A wrong
gradient in ordinary training, found by the second-derivative checks in
``lucid/test/unit/autograd/test_conv_second_derivatives.py``.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._fixtures.devices import metal_available


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_metal_transposed_input_gradient_with_output_padding_past_the_stride() -> None:
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((1, 2, 4, 5)).astype(np.float32)
    w0 = rng.standard_normal((2, 3, 3, 3)).astype(np.float32)
    grads = {}
    for device in ("cpu", "metal"):
        x = lucid.tensor(x0, device=device, requires_grad=True)
        y = F.conv_transpose2d(
            x, lucid.tensor(w0, device=device), stride=1, dilation=2, output_padding=1
        )
        (y * y).sum().backward()
        grads[device] = x.grad.numpy()
    np.testing.assert_allclose(grads["metal"], grads["cpu"], rtol=1e-4, atol=1e-4)
