"""Test class for helper methods found simon"""

import numpy as np
import pyquil.api as api
import pytest

from grove.alpha.simon.simon import (
    find_mask,
    unitary_function,
    oracle_function,
    is_unitary,
    most_significant_bit,
    check_two_to_one,
    insert_into_row_echelon_binary_matrix,
    make_square_row_echelon,
    binary_back_substitute,
    simon
)

from pyquil.quil import Program
from mock import patch, MagicMock


expected_return = [
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.]
    ]


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestFindMask(object):
    def test_one_qubit_two_to_one(self):
        _find_mask_test_helper([1, 1], 1)

    def test_two_qubits_two_to_one(self):
        _find_mask_test_helper([0, 2, 0, 2], 2)

    def test_three_qubits_two_to_one_mask_one(self):
        _find_mask_test_helper([0, 0, 7, 7, 4, 4, 2, 2], 1)

    def test_three_qubits_two_to_one_mask_five(self):
        _find_mask_test_helper([3, 0, 1, 7, 0, 3, 7, 1], 5)

    def test_four_qubits_two_to_one(self):
        _find_mask_test_helper([0, 1, 2, 3, 4, 5, 6, 7,
                                7, 6, 5, 4, 3, 2, 1, 0], 15)

    def test_one_qubit_one_to_one(self):
        _find_mask_test_helper([1, 0])

    def test_two_qubits_one_to_one(self):
        _find_mask_test_helper([0, 1, 2, 3])

    def test_three_qubits_one_to_one_odds_even(self):
        _find_mask_test_helper([1, 3, 5, 7, 0, 2, 4, 6])

    def test_three_qubits_one_to_one_random(self):
        _find_mask_test_helper([3, 0, 2, 7, 1, 6, 5, 4])

    def test_four_qubits_one_to_one(self):
        _find_mask_test_helper([9, 10, 5, 6, 12, 1, 2, 8, 14,
                                0, 13, 3, 11, 4, 7, 15])


def _find_mask_test_helper(mappings, mask=None):
    n = int(np.log2(len(mappings)))
    qvm = api.SyncConnection()

    qubits = range(n)
    ancillas = range(n, 2 * n)

    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancillas)

    s, iterations, simon_program = find_mask(qvm, oracle, qubits)
    two_to_one = check_two_to_one(qvm, oracle, ancillas, s)

    found_mask = int(s, 2)
    assert (mappings[0] == mappings[found_mask]) == two_to_one

    if two_to_one:
        assert found_mask == mask


def test_unitary_function_return():
    actual_return = unitary_function([0, 2, 2, 0])
    np.testing.assert_equal(actual_return, expected_return)


def test_oracle_program():
    func = unitary_function([0, 2, 2, 0])
    actual_prog = oracle_function(func, [0, 1], [2, 3])
    expected_prog = Program()
    expected_prog.defgate("FUNCT", expected_return)
    expected_prog.defgate("FUNCT-INV", np.linalg.inv(expected_return))
    expected_prog.inst("FUNCT 4 0 1")
    expected_prog.inst("CNOT 0 2")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("FUNCT-INV 4 0 1")
    assert expected_prog.__str__() == actual_prog.__str__()


def test_oracle_program():
    func = unitary_function([0, 2, 2, 0])
    actual_prog = simon(oracle_function(func, [0, 1], [2, 3]), [0, 1])
    expected_prog = Program()
    expected_prog.defgate("FUNCT", expected_return)
    expected_prog.defgate("FUNCT-INV", np.linalg.inv(expected_return))
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")

    expected_prog.inst("FUNCT 4 0 1")
    expected_prog.inst("CNOT 0 2")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("FUNCT-INV 4 0 1")

    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    assert expected_prog.__str__() == actual_prog.__str__()


def test_find_mask():
    func = unitary_function([0, 2, 2, 0])
    orc_func = oracle_function(func, [0, 1], [2, 3])

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.return_value = [[1, 1], [0, 1]]
    s_str, n_iter, pr = find_mask(qvm, orc_func, [0, 1])
    assert s_str == '11'
    assert n_iter == 1

    expected_prog = Program()
    expected_prog.defgate("FUNCT", expected_return)
    expected_prog.defgate("FUNCT-INV", np.linalg.inv(expected_return))
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")

    expected_prog.inst("FUNCT 4 0 1")
    expected_prog.inst("CNOT 0 2")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("FUNCT-INV 4 0 1")

    expected_prog.inst("H 0")
    expected_prog.inst("H 1")

    assert pr.__str__() == expected_prog.__str__()


def test_check_two_to_one():
    func = unitary_function([0, 2, 2, 0])
    orc_func = oracle_function(func, [0, 1], [2, 3])

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.return_value = [[1, 1], [0, 1]]
    assert check_two_to_one(qvm, orc_func, [2, 3], "11")



def test_unitary_two_by_two():
    hadamard = np.array([[1., 1.], [1., -1.]])
    hadamard *= 1 / np.sqrt(2)
    assert is_unitary(hadamard)


def test_unitary_eight_by_eight():
    matrix = np.zeros(shape=(8, 8))
    one_locations = [(0, 5), (1, 7), (2, 0), (3, 4),
                     (4, 1), (5, 2), (6, 6), (7, 3)]
    for loc in one_locations:
        matrix[loc[0], loc[1]] = 1
    assert is_unitary(matrix)


def test_not_unitary_rectangular():
    matrix = np.array([[0, 1, 0], [1, 0, 1]])
    assert not is_unitary(matrix)


def test_not_unitary_four_by_four():
    matrix = np.zeros(shape=(4, 4))
    matrix[0, 1] = 1
    matrix[1, 0] = 1
    matrix[2, 2] = 1
    matrix[3, 2] = 1
    assert not is_unitary(matrix)


def test_single_one():
    assert most_significant_bit(np.array([1])) == 0


def test_single_one_leading_zeroes():
    assert most_significant_bit(np.array([0, 1, 0, 0])) == 1


def test_multiple_ones_leading_zeroes():
    assert most_significant_bit(np.array([0, 0, 1, 1, 0, 1])) == 2


def test_no_substitution():
    W = np.array([[1, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0]])
    z = np.array([1, 1, 1, 0, 0])  # linear combination of first two rows

    W = insert_into_row_echelon_binary_matrix(W, z)

    W_expected = np.array([[1, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0]])

    assert np.allclose(W, W_expected)


def test_insert_directly():
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0]])
    z = np.array([0, 0, 1, 0, 1])

    W = insert_into_row_echelon_binary_matrix(W, z)
    W_expected = np.array([[1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 1]])

    assert np.allclose(W, W_expected)


def test_insert_after_xor():
    W = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0]])

    z = np.array([1, 0, 1, 0, 1, 1])

    W = insert_into_row_echelon_binary_matrix(W, z)
    W_expected = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1, 1]])

    assert np.allclose(W, W_expected)


def test_add_row_at_top():
    W = np.array([[0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]])
    W, insert_row_num = make_square_row_echelon(W)

    W_expected = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 1]])

    assert insert_row_num == 0

    assert np.allclose(W, W_expected)


def test_add_row_at_bottom():
    W = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0]])
    W, insert_row_num = make_square_row_echelon(W)

    W_expected = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    assert insert_row_num == 3

    assert np.allclose(W, W_expected)


def test_add_row_in_middle():
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    W, insert_row_num = make_square_row_echelon(W)

    W_expected = np.array([[1, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])

    assert insert_row_num == 1

    assert np.allclose(W, W_expected)


def test_one_at_top():
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    s = np.array([1, 0, 0, 0, 0])
    x = binary_back_substitute(W, s)

    prod = np.dot(W, x)
    prod = np.vectorize(lambda x: x % 2)(prod)

    assert np.allclose(s, prod)


def test_one_at_bottom():
    W = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    s = np.array([0, 0, 0, 1])
    x = binary_back_substitute(W, s)

    prod = np.dot(W, x)
    prod = np.vectorize(lambda x: x % 2)(prod)

    assert np.allclose(s, prod)


def test_one_at_middle():
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    s = np.array([0, 1, 0, 0, 0])
    x = binary_back_substitute(W, s)

    prod = np.dot(W, x)
    prod = np.vectorize(lambda x: x % 2)(prod)

    assert np.allclose(s, prod)
