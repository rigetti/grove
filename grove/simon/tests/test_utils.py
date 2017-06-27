"""Test class for helper methods found simon"""

import numpy as np
import pyquil.api as api
import pytest

from grove.simon.simon import find_mask, unitary_function, oracle_function, _is_unitary, _most_significant_bit


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestFindMask(object):
    def test_two_qubits(self):
        _find_mask_test_helper([0, 2, 0, 2], 2)

    def test_three_qubits_one_mask(self):
        _find_mask_test_helper([0, 0, 7, 7, 4, 4, 2, 2], 1)

    def test_three_qubits_five_mask(self):
        _find_mask_test_helper([3, 0, 1, 7, 0, 3, 7, 1], 5)

    def test_four_qubits(self):
        _find_mask_test_helper([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0], 15)


def _find_mask_test_helper(mappings, mask):
    n = int(np.log2(len(mappings)))
    qvm = api.SyncConnection()

    qubits = range(n)
    ancillas = range(n, 2*n)

    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancillas)
    
    s = find_mask(qvm, oracle, qubits)[0]

    assert int(s, 2) == mask


class TestIsUnitary(object):
    def test_unitary_two_by_two(self):
        hadamard = np.array([[1., 1.], [1., -1.]])
        hadamard *= 1/np.sqrt(2)
        assert _is_unitary(hadamard)

    def test_unitary_eight_by_eight(self):
        matrix = np.zeros(shape=(8, 8))
        one_locations = [(0, 5), (1, 7), (2, 0), (3, 4), (4, 1), (5, 2), (6, 6), (7, 3)]
        for loc in one_locations:
            matrix[loc[0], loc[1]] = 1
        assert _is_unitary(matrix)

    def test_not_unitary_rectangular(self):
        matrix = np.array([[0, 1, 0], [1, 0, 1]])
        assert not _is_unitary(matrix)

    def test_not_unitary_four_by_four(self):
        matrix = np.zeros(shape=(4, 4))
        matrix[0, 1] = 1
        matrix[1, 0] = 1
        matrix[2, 2] = 1
        matrix[3, 2] = 1
        assert not _is_unitary(matrix)


class TestMostSignificantBits(object):
    def test_single_one(self):
        assert _most_significant_bit([1]) == 0

    def test_single_one_leading_zeroes(self):
        assert _most_significant_bit([0, 1, 0, 0]) == 1

    def test_multiple_ones_leading_zeroes(self):
        assert _most_significant_bit([0, 0, 1, 1, 0, 1]) == 2
