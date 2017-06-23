"""Test class for helper methods found simon"""

from grove.simon.simon import *


class TestIsUnitary(object):
    def test_unitary_two_by_two(self):
        hadamard = np.array([[1., 1.], [1., -1.]])
        hadamard *= 1/np.sqrt(2)
        assert is_unitary(hadamard)

    def test_unitary_eight_by_eight(self):
        matrix = np.zeros(shape=(8, 8))
        one_locations = [(0, 5), (1, 7), (2, 0), (3, 4), (4, 1), (5, 2), (6, 6), (7, 3)]
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
