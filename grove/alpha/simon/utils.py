import numpy as np


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_unitary(matrix):
    """
    A helper function that checks if a matrix is unitary.

    :param 2darray matrix: a matrix to test unitarity of
    :return: true if and only if matrix is unitary
    :rtype: bool
    """
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))


def most_significant_bit(lst):
    """
    A helper function that finds the position of
    the most significant bit in a 1darray of 1s and 0s,
    i.e. the first position where a 1 appears, reading left to right.

    :param 1darray lst: a 1darray of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    :rtype: int
    """
    return np.argwhere(np.asarray(lst) == 1)[0][0]
