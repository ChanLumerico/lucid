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


import numpy as np
import torch
import torch.nn.functional as F


def winograd_conv2d_numpy(x, weight, bias=None, padding=(0, 0)):
    """
    Winograd F(2x2, 3x3) convolution for NCHW inputs using NumPy.
    Supports multichannel, batch, odd spatial dims and arbitrary padding.
    """
    N, C_in, H, W = x.shape
    C_out, _, kh, kw = weight.shape
    pad_h, pad_w = padding
    assert (kh, kw) == (3, 3), "Kernel must be 3x3"

    # 1) Pad input
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    H_out = H + 2 * pad_h - kh + 1
    W_out = W + 2 * pad_w - kw + 1

    # 2) Winograd params
    m, r = 2, 3
    alpha = m + r - 1  # = 4
    nH = int(np.ceil(H_out / m))
    nW = int(np.ceil(W_out / m))

    # 3) Pad to full tiles if needed
    H_pad = nH * m + r - 1
    W_pad = nW * m + r - 1
    extra_h = H_pad - (H + 2 * pad_h)
    extra_w = W_pad - (W + 2 * pad_w)
    if extra_h > 0 or extra_w > 0:
        x_pad = np.pad(
            x_pad, ((0, 0), (0, 0), (0, extra_h), (0, extra_w)), mode="constant"
        )

    # 4) Transform matrices
    B = np.array(
        [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=x.dtype
    )
    G = np.array(
        [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=x.dtype
    )
    A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=x.dtype)

    # 5) Transform filters
    U = np.einsum("ik, ockl, jl -> ocij", G, weight, G)

    # 6) Allocate output in tile space
    Y = np.zeros((N, C_out, nH * m, nW * m), dtype=x.dtype)

    # 7) Compute per-tile
    for i in range(nH):
        for j in range(nW):
            d = x_pad[
                :, :, i * m : i * m + alpha, j * m : j * m + alpha
            ]  # (N, C_in,4,4)
            d_flat = d.reshape(-1, alpha, alpha)  # (N*C_in,4,4)
            V_flat = B @ d_flat @ B.T  # (N*C_in,4,4)
            V = V_flat.reshape(N, C_in, alpha, alpha)  # (N,C_in,4,4)

            M = np.einsum("ocij, ncij -> noij", U, V)  # (N,C_out,4,4)

            M_flat = M.reshape(-1, alpha, alpha)  # (N*C_out,4,4)
            tmp = np.tensordot(A, M_flat, axes=(1, 1))  # (2,N*C_out,4)
            tmp = np.tensordot(tmp, A, axes=(2, 1))  # (2,N*C_out,2)
            Y_tile = tmp.transpose(1, 0, 2).reshape(N, C_out, m, m)

            Y[:, :, i * m : (i + 1) * m, j * m : (j + 1) * m] = Y_tile

    # 8) Crop and add bias
    Y = Y[:, :, :H_out, :W_out]
    if bias is not None:
        Y += bias.reshape(1, -1, 1, 1)

    return Y


# ----------------------------
# Comparison with PyTorch
# ----------------------------

# 1) Create random test data
np.random.seed(42)
N, C_in, H, weight = 4, 3, 7, 5
C_out = 6

x_np = np.random.randn(N, C_in, H, weight).astype(np.float32)
W_np = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
b_np = np.random.randn(C_out).astype(np.float32)
padding = (1, 0)

# 2) NumPy Winograd conv
out_np = winograd_conv2d_numpy(x_np, W_np, b_np, padding=padding)

# 3) PyTorch reference conv
x_t = torch.from_numpy(x_np)
W_t = torch.from_numpy(W_np)
b_t = torch.from_numpy(b_np)
out_t = F.conv2d(x_t, W_t, bias=b_t, padding=padding).numpy()

# 4) Compare
diff = np.max(np.abs(out_np - out_t))
print(f"Max absolute difference: {diff:.6e}")
assert diff < 1e-4, "Outputs differ more than tolerance!"

print("Success: NumPy Winograd matches PyTorch conv2d.")
