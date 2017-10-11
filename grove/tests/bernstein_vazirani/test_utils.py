import pytest

from grove.bernstein_vazirani import utils as u


def test_bitwise_dot_product():
    with pytest.raises(ValueError):
        _ = u.bitwise_dot_product('11', '1111')

    assert u.bitwise_dot_product('110', '110') == '0'
    assert u.bitwise_dot_product('110', '101') == '1'
    assert u.bitwise_dot_product('010', '101') == '0'


def test_bit_masking():
    with pytest.raises(ValueError):
        _ = u.bitwise_xor('11', '1111')

    bit_string = '101'
    mask_string = '110'
    assert u.bitwise_xor(bit_string, mask_string) == '011'
