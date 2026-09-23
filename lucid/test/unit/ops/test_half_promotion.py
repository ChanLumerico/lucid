"""float16 and bfloat16 meet at float32; either meets float32 at float32.

bfloat16 was missing from the promotion table, so it ranked as float32 and
whichever operand came first won: ``cat([bf16, f32])`` came back bfloat16,
dropping the float32 operand's precision without a word.
"""

import lucid


def _t(dtype: lucid.dtype) -> lucid.Tensor:
    return lucid.tensor([1.0, 2.0]).to(dtype)


def test_bfloat16_against_float32_is_float32() -> None:
    assert (_t(lucid.bfloat16) + _t(lucid.float32)).dtype == lucid.float32
    assert (_t(lucid.float32) + _t(lucid.bfloat16)).dtype == lucid.float32
    assert lucid.cat([_t(lucid.bfloat16), _t(lucid.float32)]).dtype == lucid.float32


def test_float16_against_bfloat16_is_float32() -> None:
    assert (_t(lucid.float16) + _t(lucid.bfloat16)).dtype == lucid.float32
    assert lucid.add(_t(lucid.bfloat16), _t(lucid.float16)).dtype == lucid.float32


def test_a_half_type_against_itself_stays() -> None:
    assert (_t(lucid.bfloat16) + _t(lucid.bfloat16)).dtype == lucid.bfloat16
    assert (_t(lucid.float16) + _t(lucid.float16)).dtype == lucid.float16
