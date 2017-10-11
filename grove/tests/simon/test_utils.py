import numpy as np

from grove.simon import utils as u


def test_unitary_two_by_two():
    hadamard = np.array([[1., 1.], [1., -1.]])
    hadamard *= 1 / np.sqrt(2)
    assert u.is_unitary(hadamard)


def test_unitary_eight_by_eight():
    matrix = np.zeros(shape=(8, 8))
    one_locations = [(0, 5), (1, 7), (2, 0), (3, 4),
                     (4, 1), (5, 2), (6, 6), (7, 3)]
    for loc in one_locations:
        matrix[loc[0], loc[1]] = 1
    assert u.is_unitary(matrix)


def test_not_unitary_rectangular():
    matrix = np.array([[0, 1, 0], [1, 0, 1]])
    assert not u.is_unitary(matrix)


def test_not_unitary_four_by_four():
    matrix = np.zeros(shape=(4, 4))
    matrix[0, 1] = 1
    matrix[1, 0] = 1
    matrix[2, 2] = 1
    matrix[3, 2] = 1
    assert not u.is_unitary(matrix)


def test_single_one():
    assert u.most_significant_bit(np.array([1])) == 0


def test_single_one_leading_zeroes():
    assert u.most_significant_bit(np.array([0, 1, 0, 0])) == 1


def test_multiple_ones_leading_zeroes():
    assert u.most_significant_bit(np.array([0, 0, 1, 1, 0, 1])) == 2


def test_bit_masking():
    bit_string = '101'
    mask_string = '110'
    assert u.bitwise_xor(bit_string, mask_string) == '011'


def test_binary_back_step():
    A = np.asarray([[1, 1, 0, 1],
                    [0, 1, 1, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]])
    s = np.asarray([0, 0, 0, 1])

    np.testing.assert_equal(
        u.binary_back_substitute(A, s),
        [1, 1, 0, 1]
    )

    A = np.asarray([[1, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]])
    s = np.asarray([0, 0, 0, 1])

    np.testing.assert_equal(
        u.binary_back_substitute(A, s),
        [1, 1, 0, 0]
    )

    A = np.asarray([[1, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]])
    s = np.asarray([0, 0, 0, 1])

    np.testing.assert_equal(
        u.binary_back_substitute(A, s),
        [1, 1, 0, 0]
    )

    A = np.asarray([[1, 0, 1],
                    [0, 1, 0],
                    [0, 0, 1]])
    s = np.asarray([0, 1, 0])

    np.testing.assert_equal(
        u.binary_back_substitute(A, s),
        [0, 1, 0]
    )

    A = np.asarray([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 1]])
    s = np.asarray([0, 0, 1])

    np.testing.assert_equal(
        u.binary_back_substitute(A, s),
        [1, 1, 0]
    )
