# winograd_torch_comparison.py
import numpy as np
import torch
import torch.nn.functional as F

# Adjust import according to your module path
from lucid.nn.functional._conv import _im2col_conv, _winograd_conv
from lucid._tensor import Tensor


# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Test parameters: odd spatial dims and arbitrary symmetric padding
N, C_in, H, W = 2, 3, 7, 9
C_out = 4
pad_h, pad_w = 2, 3

# Generate random input, weights, and bias
x_np = np.random.randn(N, C_in, H, W).astype(np.float32)
w_np = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
b_np = np.random.randn(C_out).astype(np.float32)

# PyTorch reference convolution
x_t = torch.from_numpy(x_np)
w_t = torch.from_numpy(w_np)
b_t = torch.from_numpy(b_np)
y_ref = F.conv2d(
    x_t, w_t, bias=b_t, stride=1, padding=(pad_h, pad_w), dilation=1, groups=1
).numpy()

# Lucid Winograd convolution
x_l = Tensor(x_np)
w_l = Tensor(w_np)
b_l = Tensor(b_np)

# y_l = _im2col_conv(x_l, w_l, b_l, (1, 1), (pad_h, pad_w), (1, 1)).data
y_l = _winograd_conv(x_l, w_l, b_l, (pad_h, pad_w)).data

# Compare outputs
diff = np.abs(y_ref - y_l)
max_err = diff.max()
mean_err = diff.mean()
print(f"Max error: {max_err:.6e}, Mean error: {mean_err:.6e}")
