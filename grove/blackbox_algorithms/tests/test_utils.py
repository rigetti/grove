"""Test class for methods found in the black box utils."""

from grove.blackbox_algorithms.utils import *
import pyquil.quil as pq
from pyquil.resource_manager import Qubit
import unittest


class TestUnitaryFromFunction(unittest.TestCase):
    def test_domain_n_to_range_one(self):
        num_domain_bits = 3
        num_range_bits = 1
        mappings = [1, 0, 0, 1, 0, 0, 1, 1]
        f = lambda x: mappings[x]
        U, new_bits = unitary_from_function(f, num_domain_bits, num_range_bits)
        self.assertTrue(is_unitary(U))
        self.assertEquals(U.shape[0], 2**(num_domain_bits + new_bits))
        self._check_consistent(f, U, num_domain_bits, num_range_bits)

    def test_domain_n_to_range_n(self):
        num_domain_bits = 3
        num_range_bits = 3
        mappings = [0, 1, 2, 3, 3, 2, 1, 0]
        f = lambda x: mappings[x]
        U, new_bits = unitary_from_function(f, num_domain_bits, num_range_bits)
        self.assertTrue(is_unitary(U))
        self.assertEquals(U.shape[0], 2**(num_domain_bits + new_bits))
        self._check_consistent(f, U, num_domain_bits, num_range_bits)

    def test_domain_one_to_range_n(self):
        num_domain_bits = 2
        num_range_bits = 4
        mappings = [0, 4, 1, 7]
        f = lambda x: mappings[x]
        U, new_bits = unitary_from_function(f, num_domain_bits, num_range_bits)
        self.assertTrue(is_unitary(U))
        self.assertEquals(U.shape[0], 2**(num_domain_bits + new_bits))
        self._check_consistent(f, U, num_domain_bits, num_range_bits)

    def _check_consistent(self, f, U, num_domain_bits, num_range_bits):
        for x in range(2**num_domain_bits):
            y_expected = f(x)

            input = np.zeros(U.shape[0])
            input[x] = 1
            output = np.dot(U, input)
            one_index = -1
            for i in range(len(output)):
                if output[i] == 1:
                    self.assertEqual(one_index, -1)
                    one_index = i
                else:
                    self.assertEqual(output[i], 0)

            self.assertNotEqual(one_index, -1)
            self.assertEqual(one_index % 2**num_range_bits, y_expected)

class TestIntegerToBitstring(unittest.TestCase):
    def test_integer_five(self):
        self.assertEqual(integer_to_bitstring(5, 3), '101')

    def test_integer_fourteen(self):
        self.assertEqual(integer_to_bitstring(14, 4), '1110')

    def test_integer_zero(self):
        self.assertEqual(integer_to_bitstring(0, 1), '0')

    def test_integer_three_with_leading_zeros(self):
        self.assertEqual(integer_to_bitstring(3, 6), '000011')

    def test_integer_eleven_truncated(self):
        self.assertEquals(integer_to_bitstring(11, 3), '011')


class TestBitstringToArray(unittest.TestCase):
    def test_one_bit_bitstring(self):
        self.assertEqual(list(bitstring_to_array('1')), [1])

    def test_three_bit_bitstring(self):
        self.assertEqual(list(bitstring_to_array('100')), [1, 0, 0])

    def test_six_bit_bitstring(self):
        self.assertEqual(list(bitstring_to_array('001101')), [0, 0, 1, 1, 0, 1])


class TestBitstringToInteger(unittest.TestCase):
    def test_three_bit_bitstring_all_ones(self):
        self.assertEqual(bitstring_to_integer('111'), 7)

    def test_one_bit_bitstring(self):
        self.assertEqual(bitstring_to_integer('1'), 1)

    def test_five_bit_bitstring_ones_and_zeros(self):
        self.assertEqual(bitstring_to_integer('10010'), 18)

    def test_leading_zeros_bitstring(self):
        self.assertEqual(bitstring_to_integer('001101'), 13)


class TestIsUnitary(unittest.TestCase):
    def test_unitary_two_by_two(self):
        hadamard = np.array([[1., 1.], [1., -1.]])
        hadamard *= 1/np.sqrt(2)
        self.assertTrue(is_unitary(hadamard))

    def test_unitary_eight_by_eight(self):
        matrix = np.zeros(shape=(8, 8))
        one_locations = [(0, 5), (1, 7), (2, 0), (3, 4), (4, 1), (5, 2), (6, 6), (7, 3)]
        for loc in one_locations:
            matrix[loc[0], loc[1]] = 1
        self.assertTrue(is_unitary(matrix))

    def test_not_unitary_rectangular(self):
        matrix = np.array([[0, 1, 0], [1, 0, 1]])
        self.assertFalse(is_unitary(matrix))

    def test_not_unitary_four_by_four(self):
        matrix = np.zeros(shape=(4, 4))
        matrix[0, 1] = 1
        matrix[1, 0] = 1
        matrix[2, 2] = 1
        matrix[3, 2] = 1
        self.assertFalse(is_unitary(matrix))


class TestGetNBits(unittest.TestCase):
    def test_get_one_bit(self):
        p = pq.Program()
        bits = get_n_bits(p, 1)
        self.assertEqual(len(bits), 1)
        self.assertEqual(type(bits[0]), Qubit)

    def test_get_three_bits(self):
        p = pq.Program()
        bits = get_n_bits(p, 3)
        self.assertEqual(len(bits), 3)
        self.assertEqual(type(bits[0]), Qubit)
        self.assertEqual(type(bits[1]), Qubit)
        self.assertEqual(type(bits[2]), Qubit)


class TestMostSignificantBits(unittest.TestCase):
    def test_single_one(self):
        self.assertEqual(most_significant_bit([1]), 0)

    def test_single_one_leading_zeroes(self):
        self.assertEqual(most_significant_bit([0, 1, 0, 0]), 1)

    def test_multiple_ones_leading_zeroes(self):
        self.assertEqual(most_significant_bit([0, 0, 1, 1, 0, 1]), 2)

if __name__ == '__main__':
    unittest.main()
