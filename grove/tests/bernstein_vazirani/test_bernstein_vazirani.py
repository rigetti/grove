import numpy as np
from mock import patch

from grove.bernstein_vazirani.bernstein_vazirani import BernsteinVazirani, create_bv_bitmap


def test_bv_bitmap_generator():
    expected_bit_map = {
        '000': '1',
        '001': '1',
        '010': '0',
        '011': '0',
        '100': '0',
        '101': '0',
        '110': '1',
        '111': '1'
    }
    a = '110'
    b = '1'
    actual_bitmap = create_bv_bitmap(a, b)

    assert actual_bitmap == expected_bit_map


def test_bv_class_with_bitmap():
    bit_map = {
        '000': '1',
        '001': '1',
        '010': '0',
        '011': '0',
        '100': '0',
        '101': '0',
        '110': '1',
        '111': '1'
    }

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.side_effect = [
            ([0, 1, 1], ),
            ([1], )
        ]

    bv = BernsteinVazirani()
    bv.run(qvm, bit_map).check_solution()
    bv_a, bv_b = bv.solution
    assert bv_a == '110'
    assert bv_b == '1'


def test_bv_class_with_check_results():
    a = '1011'
    b = '0'

    bit_map = create_bv_bitmap(a, b)

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.side_effect = [
            ([1, 1, 0, 1], ),
            ([0], )
        ]

    bv = BernsteinVazirani()
    bv.run(qvm, bit_map).check_solution()
    bv_a, bv_b = bv.solution
    assert bv_a == a
    assert bv_b == b


def test_bv_class_with_return_solution():
    a = '1011'
    b = '0'

    bit_map = create_bv_bitmap(a, b)

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.side_effect = [
            ([1, 1, 0, 1], ),
            ([0], )
        ]

    bv = BernsteinVazirani()
    bv_a, bv_b = bv.run(qvm, bit_map).get_solution()
    assert bv_a == a
    assert bv_b == b


def test_bv_unitary_generator():
    expected_transition_dct = {
        '000': '100',
        '001': '101',
        '010': '010',
        '011': '011',
        '100': '000',
        '101': '001',
        '110': '110',
        '111': '111'
    }
    expected_unitary = [
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.]
    ]

    bit_map = {
        '00': '1',
        '01': '1',
        '10': '0',
        '11': '0',
    }
    actual_unitary, actual_dct = BernsteinVazirani()._compute_unitary_oracle_matrix(bit_map)
    assert expected_transition_dct == actual_dct
    np.testing.assert_equal(actual_unitary, expected_unitary)
