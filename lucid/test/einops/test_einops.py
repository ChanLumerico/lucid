import numpy as np
import torch

import lucid
from lucid.test._utils.test_utils import _UtilsTorchOpBase, _unwrap_callable
from lucid.test.core import TensorOpTorchCase


class _EinopsTorchBase(_UtilsTorchOpBase):
    _collect_nested_test_classes = True

    def _build_case(self) -> TensorOpTorchCase:
        forward_raw = getattr(type(self), "forward_op", None)
        torch_raw = getattr(type(self), "torch_forward", None)

        forward_op = _unwrap_callable(forward_raw)
        torch_forward = None if torch_raw is None else _unwrap_callable(torch_raw)

        return TensorOpTorchCase(
            name=self.case_name,
            build_inputs=lambda _: self._build_inputs(),
            forward_op=forward_op,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=self._normalize_bool_flags(
                self.requires_grad, len(self.inputs)
            ),
            check_finite=self.check_finite,
            rtol=self.rtol,
            atol=self.atol,
            torch_forward=torch_forward,
            compare_grad_with_torch=self.compare_grad_with_torch,
        )


class TestRearrange(_EinopsTorchBase):
    __test__ = False

    class TestTranspose(_EinopsTorchBase):
        case_name = "rearrange_transpose"
        input = np.arange(6, dtype=np.float64).reshape(2, 3)
        inputs = (input,)
        forward_op = staticmethod(lambda a: lucid.einops.rearrange(a, "b c -> c b"))
        torch_forward = staticmethod(lambda a: torch.permute(a, (1, 0)))
        expected_forward = input.T

    class TestMergeDims(_EinopsTorchBase):
        case_name = "rearrange_merge_dims"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.rearrange(a, "b c d -> b (c d)")
        )
        torch_forward = staticmethod(lambda a: torch.reshape(a, (2, 12)))
        expected_forward = input.reshape(2, 12)

    class TestSplitKnown(_EinopsTorchBase):
        case_name = "rearrange_split_known"
        input = np.arange(24, dtype=np.float64).reshape(6, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.rearrange(a, "(b c) d -> b c d", b=2, c=3)
        )
        torch_forward = staticmethod(lambda a: torch.reshape(a, (2, 3, 4)))
        expected_forward = input.reshape(2, 3, 4)

    class TestFlattenNamed(_EinopsTorchBase):
        case_name = "rearrange_flatten_named"
        input = np.arange(24, dtype=np.float64).reshape(6, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.rearrange(a, "(a b) c -> a b c", a=2, b=3)
        )
        torch_forward = staticmethod(lambda a: torch.reshape(a, (2, 3, 4)))
        expected_forward = input.reshape(2, 3, 4)

    class TestEllipsis(_EinopsTorchBase):
        case_name = "rearrange_ellipsis"
        input = np.arange(120, dtype=np.float64).reshape(2, 3, 4, 5)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.rearrange(a, "... c d -> d ... c")
        )
        torch_forward = staticmethod(lambda a: torch.permute(a, (3, 0, 1, 2)))
        expected_forward = np.transpose(input, (3, 0, 1, 2))

    class TestFlattenAll(_EinopsTorchBase):
        case_name = "rearrange_flatten_all"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.rearrange(a, "a b c -> (a b c)")
        )
        torch_forward = staticmethod(lambda a: torch.reshape(a, (24,)))
        expected_forward = input.reshape(-1)


class TestReduce(_EinopsTorchBase):
    __test__ = False

    class TestSumOverTwoAxes(_EinopsTorchBase):
        case_name = "reduce_sum_two_axes"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.reduce(a, "b c d -> b", reduction="sum")
        )
        torch_forward = staticmethod(lambda a: torch.sum(a, dim=(1, 2)))
        expected_forward = input.sum(axis=(1, 2))

    class TestMeanAxis(_EinopsTorchBase):
        case_name = "reduce_mean_axis"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.reduce(a, "b c d -> b d", reduction="mean")
        )
        torch_forward = staticmethod(lambda a: torch.mean(a, dim=1))
        expected_forward = input.mean(axis=1)

    class TestKeepAxis(_EinopsTorchBase):
        case_name = "reduce_keep_axis"
        input = np.arange(24, dtype=np.float64).reshape(6, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.reduce(a, "(b c) d -> b", reduction="mean", b=2, c=3)
        )
        torch_forward = staticmethod(
            lambda a: torch.mean(a.reshape(2, 3, 4), dim=(1, 2))
        )
        expected_forward = input.reshape(2, 3, 4).mean(axis=(1, 2))

    class TestTupleOutput(_EinopsTorchBase):
        case_name = "reduce_tuple_output"
        input = np.arange(48, dtype=np.float64).reshape(6, 8)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.reduce(
                a, "(b c) d -> (b d)", reduction="sum", b=2, c=3
            )
        )
        torch_forward = staticmethod(
            lambda a: torch.sum(a.reshape(2, 3, 8), dim=1).reshape(-1)
        )
        expected_forward = input.reshape(2, 3, 8).sum(axis=1).reshape(-1)

    class TestEllipsis(_EinopsTorchBase):
        case_name = "reduce_ellipsis"
        input = np.arange(30, dtype=np.float64).reshape(2, 3, 5)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.reduce(a, "... c -> ...", reduction="sum")
        )
        torch_forward = staticmethod(lambda a: torch.sum(a, dim=-1))
        expected_forward = input.sum(axis=2)

    class TestReduceAll(_EinopsTorchBase):
        case_name = "reduce_all"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.reduce(a, "a b c ->", reduction="sum")
        )
        torch_forward = staticmethod(lambda a: torch.sum(a))
        expected_forward = input.sum()


class TestRepeat(_EinopsTorchBase):
    __test__ = False

    class TestNewAxis(_EinopsTorchBase):
        case_name = "repeat_new_axis"
        input = np.arange(8, dtype=np.float64).reshape(2, 4)
        inputs = (input,)
        forward_op = staticmethod(lambda a: lucid.einops.repeat(a, "b c -> b c n", n=2))
        torch_forward = staticmethod(lambda a: a.unsqueeze(-1).repeat(1, 1, 2))
        expected_forward = np.repeat(input[:, :, None], repeats=2, axis=2)

    class TestTwoNewAxes(_EinopsTorchBase):
        case_name = "repeat_two_new_axes"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.repeat(a, "b c d -> b c d n m", n=1, m=2)
        )
        torch_forward = staticmethod(
            lambda a: a.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, 2)
        )
        expected_forward = np.repeat(input[:, :, :, None, None], repeats=2, axis=4)

    class TestPermuteAndRepeat(_EinopsTorchBase):
        case_name = "repeat_permute_and_repeat"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.repeat(a, "b c d -> d b c n", n=2)
        )
        torch_forward = staticmethod(
            lambda a: torch.permute(a, (2, 0, 1)).unsqueeze(-1).repeat(1, 1, 1, 2)
        )
        expected_forward = np.repeat(
            np.transpose(input, (2, 0, 1))[:, :, :, None], repeats=2, axis=3
        )

    class TestEllipsis(_EinopsTorchBase):
        case_name = "repeat_ellipsis"
        input = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.repeat(a, "... c -> ... c n", n=2)
        )
        torch_forward = staticmethod(
            lambda a: a.unsqueeze(-1).repeat_interleave(2, dim=-1)
        )
        expected_forward = np.repeat(input[..., None], repeats=2, axis=3)

    class TestSingletonSplit(_EinopsTorchBase):
        case_name = "repeat_singleton_split"
        input = np.arange(8, dtype=np.float64).reshape(2, 4)
        inputs = (input,)
        forward_op = staticmethod(
            lambda a: lucid.einops.repeat(a, "b c -> b (c n)", n=2)
        )
        torch_forward = staticmethod(lambda a: a.unsqueeze(-1))
        expected_forward = input[:, :, None]


class TestEinsum(_EinopsTorchBase):
    __test__ = False

    class TestOuter(_EinopsTorchBase):
        case_name = "einsum_outer"
        left = np.arange(6, dtype=np.float64).reshape(2, 3)
        right = np.arange(8, dtype=np.float64).reshape(2, 4)
        inputs = (left, right)
        forward_op = staticmethod(lambda a, b: lucid.einops.einsum("bi,bj->bij", a, b))
        torch_forward = staticmethod(lambda a, b: torch.einsum("bi,bj->bij", a, b))
        expected_forward = np.einsum("bi,bj->bij", left, right)

    class TestMatMul(_EinopsTorchBase):
        case_name = "einsum_matmul"
        left = np.arange(12, dtype=np.float64).reshape(3, 4)
        right = np.arange(20, dtype=np.float64).reshape(4, 5)
        inputs = (left, right)
        forward_op = staticmethod(lambda a, b: lucid.einops.einsum("ij,jk->ik", a, b))
        torch_forward = staticmethod(lambda a, b: torch.einsum("ij,jk->ik", a, b))
        expected_forward = np.einsum("ij,jk->ik", left, right)

    class TestTrace(_EinopsTorchBase):
        case_name = "einsum_trace"
        input = np.arange(9, dtype=np.float64).reshape(3, 3)
        inputs = (input,)
        forward_op = staticmethod(lambda a: lucid.einops.einsum("ii->", a))
        torch_forward = staticmethod(lambda a: torch.einsum("ii->", a))
        expected_forward = np.einsum("ii->", input)
        compare_grad_with_torch = False

    class TestIdentity(_EinopsTorchBase):
        case_name = "einsum_identity"
        input = np.arange(6, dtype=np.float64).reshape(2, 3)
        inputs = (input,)
        forward_op = staticmethod(lambda a: lucid.einops.einsum("b c", a))
        torch_forward = staticmethod(lambda a: torch.einsum("b c", a))
        expected_forward = input
        compare_grad_with_torch = False

    class TestImplicitMatrix(_EinopsTorchBase):
        case_name = "einsum_implicit_matrix"
        left = np.arange(6, dtype=np.float64).reshape(2, 3)
        right = np.arange(8, dtype=np.float64).reshape(2, 4)
        inputs = (left, right)
        forward_op = staticmethod(lambda a, b: lucid.einops.einsum("bi,bj", a, b))
        torch_forward = staticmethod(lambda a, b: torch.einsum("bi,bj", a, b))
        expected_forward = np.einsum("bi,bj", left, right)

    class TestImplicitScalar(_EinopsTorchBase):
        case_name = "einsum_implicit_scalar"
        left = np.arange(6, dtype=np.float64).reshape(2, 3)
        right = np.arange(6, dtype=np.float64).reshape(2, 3)
        inputs = (left, right)
        forward_op = staticmethod(lambda a, b: lucid.einops.einsum("bi,bi", a, b))
        torch_forward = staticmethod(lambda a, b: torch.einsum("bi,bi", a, b))
        expected_forward = np.einsum("bi,bi", left, right)

    class TestChainImplicit(_EinopsTorchBase):
        case_name = "einsum_chain_implicit"
        left = np.arange(4, dtype=np.float64).reshape(2, 2)
        middle = np.arange(6, dtype=np.float64).reshape(2, 3)
        right = np.arange(6, dtype=np.float64).reshape(3, 2)
        inputs = (left, middle, right)
        forward_op = staticmethod(
            lambda a, b, c: lucid.einops.einsum("b i,i j,j k", a, b, c)
        )
        torch_forward = staticmethod(
            lambda a, b, c: torch.einsum("b i,i j,j k", a, b, c)
        )
        expected_forward = np.einsum("b i,i j,j k", left, middle, right)

    class TestBatchMatMul(_EinopsTorchBase):
        case_name = "einsum_batch_matmul"
        left = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        right = np.arange(16, dtype=np.float64).reshape(2, 4, 2)
        inputs = (left, right)
        forward_op = staticmethod(
            lambda a, b: lucid.einops.einsum("b i j,b j k->b i k", a, b)
        )
        torch_forward = staticmethod(
            lambda a, b: torch.einsum("b i j,b j k->b i k", a, b)
        )
        expected_forward = np.einsum("b i j,b j k->b i k", left, right)
