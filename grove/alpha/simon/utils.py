import numpy as np
from operator import xor

PADDED_BINARY_BIT_STRING = "{0:0{1:0d}b}"


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


def bitwise_xor(bs0, bs1):
    """
    A helper to calculate the bitwise XOR of two bit string

    :param bs0: String of 0's and 1's representing a number in binary representations
    :param bs1: String of 0's and 1's representing a number in binary representations
    :return: String of 0's and 1's representing the XOR between bs0 and bs1
    :rtype: str
    """
    if len(bs0) != len(bs1):
        raise ValueError("Bit strings are not of equal length")
    n_bits = len(bs0)
    return PADDED_BINARY_BIT_STRING.format(xor(int(bs0, 2), int(bs1, 2)), n_bits)


def binary_back_substitute(W, s):
    """
    Perform back substitution on a binary system of equations, i.e. it performs Gauss elimination
    over the field :math:`GF(2)`. It finds an :math:`\\mathbf{x}` such that
    :math:`\\mathbf{\\mathit{W}}\\mathbf{x}=\\mathbf{s}`, where all arithmetic is taken bitwise
    and modulo 2.

    :param 2darray W: A square :math:`n\\times n` matrix of 0s and 1s,
              in row-echelon (upper-triangle) form
    :param 1darray s: An :math:`n\\times 1` vector of 0s and 1s
    :return: The :math:`n\\times 1` vector of 0s and 1s that solves the above
             system of equations.
    :rtype: 1darray
    """
    # iterate backwards, starting from second to last row for back-substitution
    m = np.copy(s)
    n = len(s)
    for row_num in range(n - 2, -1, -1):
        row = W[row_num]
        for col_num in range(row_num + 1, n):
            if row[col_num] == 1:
                m[row_num] = xor(s[row_num], s[col_num])

    return m[::-1]
