import numpy as np
import torch
import torch.nn.functional as F

G = np.array(
    [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=np.float32
)

B = np.array(
    [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32
)

A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)


def winograd_f23_multi(input, kernel, bias=None, padding=(0, 0)):
    if isinstance(padding, int):
        pad_h1 = pad_w1 = padding
    else:
        pad_h1, pad_w1 = padding

    batched = input.ndim == 4
    if not batched:
        input = input[None, ...]

    N = input.shape[0]
    C_out = kernel.shape[0]

    inp = np.pad(
        input, ((0, 0), (0, 0), (pad_h1, pad_h1), (pad_w1, pad_w1)), mode="constant"
    )

    H1, W1 = inp.shape[2], inp.shape[3]
    pad_h_min = max(0, 4 - H1)
    pad_w_min = max(0, 4 - W1)
    H2, W2 = H1 + pad_h_min, W1 + pad_w_min
    pad_h_even = (2 - ((H2 - 2) % 2)) % 2
    pad_w_even = (2 - ((W2 - 2) % 2)) % 2

    inp = np.pad(
        inp,
        ((0, 0), (0, 0), (0, pad_h_min + pad_h_even), (0, pad_w_min + pad_w_even)),
        mode="constant",
    )

    H_pad, W_pad = inp.shape[2], inp.shape[3]
    H_out_pad = H_pad - 2
    W_out_pad = W_pad - 2
    tiles_r = H_out_pad // 2
    tiles_c = W_out_pad // 2

    U = np.einsum("ij,ocjk,kl->ocil", G, kernel, G.T)

    out_pad = np.zeros((N, C_out, H_out_pad, W_out_pad), dtype=input.dtype)

    for n in range(N):
        for i in range(tiles_r):
            for j in range(tiles_c):
                d = inp[n, :, i * 2 : i * 2 + 4, j * 2 : j * 2 + 4]
                V = np.einsum("ij,cjk,kl->cil", B, d, B.T)
                M = np.einsum("ocij,cij->oij", U, V)
                for o in range(C_out):
                    out_pad[n, o, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = A @ M[o] @ A.T

    H_out = H1 - 2
    W_out = W1 - 2
    out = out_pad[:, :, :H_out, :W_out]

    if bias is not None:
        b = bias.reshape(1, C_out, 1, 1)
        out = out + b

    if not batched:
        out = out[0]
    return out


if __name__ == "__main__":
    np.random.seed(0)
    inp_np = np.random.randn(2, 3, 7, 9).astype(np.float32)
    kern_np = np.random.randn(4, 3, 3, 3).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)

    for pad in [(0, 0), (3, 3), (3, 1)]:
        out_win = winograd_f23_multi(inp_np, kern_np, bias_np, padding=pad)
        out_tch = F.conv2d(
            torch.from_numpy(inp_np),
            torch.from_numpy(kern_np),
            bias=torch.from_numpy(bias_np),
            padding=pad,
        ).numpy()
        diff = np.max(np.abs(out_win - out_tch))
        print(f"padding={pad}, max diff={diff:.6e}")
        assert diff < 1e-5, f"Mismatch at padding={pad}: {diff}"
