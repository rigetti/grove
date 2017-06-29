"""Test class for helper methods found simon"""

import numpy as np
import pyquil.api as api
import pytest

from grove.simon.simon import find_mask, unitary_function, \
    oracle_function, is_unitary, most_significant_bit, check_two_to_one, \
    insert_into_binary_matrix, make_square_row_echelon


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


class TestIsUnitary(object):
    def test_unitary_two_by_two(self):
        hadamard = np.array([[1., 1.], [1., -1.]])
        hadamard *= 1 / np.sqrt(2)
        assert is_unitary(hadamard)

    def test_unitary_eight_by_eight(self):
        matrix = np.zeros(shape=(8, 8))
        one_locations = [(0, 5), (1, 7), (2, 0), (3, 4),
                         (4, 1), (5, 2), (6, 6), (7, 3)]
        for loc in one_locations:
            matrix[loc[0], loc[1]] = 1
        assert is_unitary(matrix)

    def test_not_unitary_rectangular(self):
        matrix = np.array([[0, 1, 0], [1, 0, 1]])
        assert not is_unitary(matrix)

    def test_not_unitary_four_by_four(self):
        matrix = np.zeros(shape=(4, 4))
        matrix[0, 1] = 1
        matrix[1, 0] = 1
        matrix[2, 2] = 1
        matrix[3, 2] = 1
        assert not is_unitary(matrix)


class TestMostSignificantBits(object):
    def test_single_one(self):
        assert most_significant_bit([1]) == 0

    def test_single_one_leading_zeroes(self):
        assert most_significant_bit([0, 1, 0, 0]) == 1

    def test_multiple_ones_leading_zeroes(self):
        assert most_significant_bit([0, 0, 1, 1, 0, 1]) == 2


class TestInsertIntoBinaryMatrix(object):
    def test_no_substitution(self):
        W = np.array([[1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0]])
        z = np.array([1, 1, 1, 0, 0])  # linear combination of first two rows

        W = insert_into_binary_matrix(W, z)

        W_expected = np.array([[1, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0]])

        assert np.allclose(W, W_expected)

    def test_insert_directly(self):
        W = np.array([[1, 1, 0, 0, 0],
                      [0, 1, 0, 1, 0]])
        z = np.array([0, 0, 1, 0, 1])

        W = insert_into_binary_matrix(W, z)
        W_expected = np.array([[1, 1, 0, 0, 0],
                               [0, 1, 0, 1, 0],
                               [0, 0, 1, 0, 1]])

        assert np.allclose(W, W_expected)

    def test_insert_after_xor(self):
        W = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0]])

        z = np.array([1, 0, 1, 0, 1, 1])

        W = insert_into_binary_matrix(W, z)
        W_expected = np.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 0],
                               [0, 0, 1, 0, 1, 1]])

        assert np.allclose(W, W_expected)


class TestMakeSquareRowEchelon(object):
    def test_add_row_at_top(self):
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

    def test_add_row_at_bottom(self):
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

    def test_add_row_in_middle(self):
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
