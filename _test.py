import lucid
from lucid._func._backend import create_func_op
from lucid._tensor import Tensor


@create_func_op(n_in=2, n_ret=2)
def some_op(a: Tensor, b: Tensor, *args):
    ret_a = Tensor(a.data + b.data)
    ret_b = Tensor(a.data - b.data)

    if args:
        print(args)

    def compute_grad_a():
        return ret_a.grad, ret_a.grad

    def compute_grad_b():
        return ret_b.grad, -ret_b.grad

    return (ret_a, compute_grad_a), (ret_b, compute_grad_b)


a = Tensor([1, 2], requires_grad=True)
b = Tensor([3, 4], requires_grad=True)

c = a * b
c.backward()

print(a.grad, b.grad)
