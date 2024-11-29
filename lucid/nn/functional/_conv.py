import lucid
from lucid._tensor import Tensor


def _im2col_2d(
    input_: Tensor, filter_size: tuple[int, int], stride: int = 1, padding: int = 0
) -> Tensor:
    N, C, H, W = input_.shape
    filter_h, filter_w = filter_size

    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = lucid.pad(input_, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    col = lucid.zeros(
        (N, C, filter_h, filter_w, out_h, out_w),
        requires_grad=input_.requires_grad,
        dtype=input_.dtype,
    )

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(
        N * out_h * out_w, C * filter_h * filter_w
    )
    return col


def conv2d(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    N, _, H, W = input_.shape
    C_out, _, K, _ = weight.shape

    out_h = (H + 2 * padding - K) // stride + 1
    out_w = (W + 2 * padding - K) // stride + 1

    col = _im2col_2d(input_, (K, K), stride, padding)
    weight_reshape = weight.reshape(C_out, -1)

    out = col @ weight_reshape.T
    out = out.reshape(N, out_h, out_w, C_out).transpose((0, 3, 1, 2))

    if bias is not None:
        out += bias.reshape(1, -1, 1, 1)

    return out
