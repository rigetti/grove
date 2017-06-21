"""Test class for methods found in the black box utils."""

from grove.simon.simon import *
import pyquil.quil as pq
from pyquil.resource_manager import Qubit


class TestIntegerToBitstring(object):
    def test_integer_five(self):
        assert integer_to_bitstring(5, 3) == '101'

    def test_integer_fourteen(self):
        assert integer_to_bitstring(14, 4) == '1110'

    def test_integer_zero(self):
        assert integer_to_bitstring(0, 1) == '0'

    def test_integer_three_with_leading_zeros(self):
        assert integer_to_bitstring(3, 6) == '000011'

    def test_integer_eleven_truncated(self):
        assert integer_to_bitstring(11, 3) == '011'


class TestBitstringToArray(object):
    def test_one_bit_bitstring(self):
        assert list(bitstring_to_array('1')) == [1]

    def test_three_bit_bitstring(self):
        assert list(bitstring_to_array('100')) == [1, 0, 0]

    def test_six_bit_bitstring(self):
        assert list(bitstring_to_array('001101')) == [0, 0, 1, 1, 0, 1]


class TestBitstringToInteger(object):
    def test_three_bit_bitstring_all_ones(self):
        assert bitstring_to_integer('111') == 7

    def test_one_bit_bitstring(self):
        assert bitstring_to_integer('1') == 1

    def test_five_bit_bitstring_ones_and_zeros(self):
        assert bitstring_to_integer('10010') == 18

    def test_leading_zeros_bitstring(self):
        assert bitstring_to_integer('001101') == 13


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


class TestGetNBits(object):
    def test_get_one_bit(self):
        p = pq.Program()
        bits = get_n_bits(p, 1)
        assert len(bits) == 1
        assert type(bits[0]) == Qubit

    def test_get_three_bits(self):
        p = pq.Program()
        bits = get_n_bits(p, 3)
        assert len(bits) == 3
        assert type(bits[0]) == Qubit
        assert type(bits[1]) == Qubit
        assert type(bits[2]) == Qubit


class TestMostSignificantBits(object):
    def test_single_one(self):
        assert most_significant_bit([1]) == 0

    def test_single_one_leading_zeroes(self):
        assert most_significant_bit([0, 1, 0, 0]) == 1

    def test_multiple_ones_leading_zeroes(self):
        assert most_significant_bit([0, 0, 1, 1, 0, 1]) == 2
