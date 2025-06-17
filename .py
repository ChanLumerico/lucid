# import numpy as np
# import torch
# import torch.nn.functional as F

# G = np.array(
#     [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=np.float32
# )

# B = np.array(
#     [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32
# )

# A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)


# def winograd_f23_multi(input, kernel, bias=None, padding=(0, 0)):
#     if isinstance(padding, int):
#         pad_h1 = pad_w1 = padding
#     else:
#         pad_h1, pad_w1 = padding

#     batched = input.ndim == 4
#     if not batched:
#         input = input[None, ...]

#     N = input.shape[0]
#     C_out = kernel.shape[0]

#     inp = np.pad(
#         input, ((0, 0), (0, 0), (pad_h1, pad_h1), (pad_w1, pad_w1)), mode="constant"
#     )

#     H1, W1 = inp.shape[2], inp.shape[3]
#     pad_h_min = max(0, 4 - H1)
#     pad_w_min = max(0, 4 - W1)
#     H2, W2 = H1 + pad_h_min, W1 + pad_w_min
#     pad_h_even = (2 - ((H2 - 2) % 2)) % 2
#     pad_w_even = (2 - ((W2 - 2) % 2)) % 2

#     inp = np.pad(
#         inp,
#         ((0, 0), (0, 0), (0, pad_h_min + pad_h_even), (0, pad_w_min + pad_w_even)),
#         mode="constant",
#     )

#     H_pad, W_pad = inp.shape[2], inp.shape[3]
#     H_out_pad = H_pad - 2
#     W_out_pad = W_pad - 2
#     tiles_r = H_out_pad // 2
#     tiles_c = W_out_pad // 2

#     U = np.einsum("ij,ocjk,kl->ocil", G, kernel, G.T)

#     out_pad = np.zeros((N, C_out, H_out_pad, W_out_pad), dtype=input.dtype)

#     for n in range(N):
#         for i in range(tiles_r):
#             for j in range(tiles_c):
#                 d = inp[n, :, i * 2 : i * 2 + 4, j * 2 : j * 2 + 4]
#                 V = np.einsum("ij,cjk,kl->cil", B, d, B.T)
#                 M = np.einsum("ocij,cij->oij", U, V)
#                 for o in range(C_out):
#                     out_pad[n, o, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = A @ M[o] @ A.T

#     H_out = H1 - 2
#     W_out = W1 - 2
#     out = out_pad[:, :, :H_out, :W_out]

#     if bias is not None:
#         bias = bias.reshape(1, C_out, 1, 1)
#         out = out + bias

#     if not batched:
#         out = out[0]
#     return out


# if __name__ == "__main__":
#     np.random.seed(42)
#     inp_np = np.random.randn(2, 3, 64, 64).astype(np.float32)
#     kern_np = np.random.randn(6, 3, 3, 3).astype(np.float32)
#     bias_np = np.random.randn(6).astype(np.float32)

#     for pad in [(0, 0), (3, 3), (3, 1)]:
#         out_win = winograd_f23_multi(inp_np, kern_np, bias_np, padding=pad)
#         out_tch = F.conv2d(
#             torch.from_numpy(inp_np),
#             torch.from_numpy(kern_np),
#             bias=torch.from_numpy(bias_np),
#             padding=pad,
#         ).numpy()
#         diff = np.max(np.abs(out_win - out_tch))
#         print(f"padding={pad}, max diff={diff:.6e}")
#         assert diff < 1e-5, f"Mismatch at padding={pad}: {diff}"


import torch
import torch.nn.functional as F
import numpy as np
import lucid
from lucid import Tensor

# Parameters
N, C_in, C_out, H, W = 2, 3, 4, 8, 8
kernel_size = 3
padding = 1

# PyTorch tensors (with grad)
torch_input = torch.randn(N, C_in, H, W, dtype=torch.float32, requires_grad=True)
torch_weight = torch.randn(C_out, C_in, kernel_size, kernel_size, requires_grad=True)
torch_bias = torch.randn(C_out, requires_grad=True)

# Forward and loss
torch_output = F.conv2d(
    torch_input, torch_weight, torch_bias, stride=1, padding=padding
)
torch_output.backward(torch.ones_like(torch_output))

# Convert to Lucid
input_ = Tensor(torch_input.detach().numpy(), requires_grad=True)
weight = Tensor(torch_weight.detach().numpy(), requires_grad=True)
bias = Tensor(torch_bias.detach().numpy(), requires_grad=True)

# Forward and backward
lucid_output = lucid.nn.functional.conv2d(
    input_, weight, bias, stride=1, padding=padding
)
lucid_output.backward()


# Compare gradients
def check_grad(name, torch_grad, lucid_tensor):
    lucid_grad = lucid_tensor.grad
    error = np.abs(torch_grad.detach().numpy() - lucid_grad).max()
    mse = np.square(torch_grad.detach().numpy() - lucid_grad).mean()
    print(f"[{name}] max abs diff: {error:.6e}, MSE: {mse:.6e}")
    assert error < 1e-4, f"{name} gradient mismatch!: {error:.6e}"
    print(f"✅ {name} gradient matches.")


check_grad("input", torch_input.grad, input_)
check_grad("weight", torch_weight.grad, weight)
check_grad("bias", torch_bias.grad, bias)
