import numpy as np

PADDED_BINARY_BIT_STRING = "{0:0{1:0d}b}"


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


def mapping_list_to_dict(lst):
    n_bits = len(lst).bit_length() - 1
    if len(lst) != 2 ** n_bits:
        raise ValueError("mappings must have a length that is a power of two")

    return {PADDED_BINARY_BIT_STRING.format(idx, n_bits):
                PADDED_BINARY_BIT_STRING.format(val, n_bits) for idx, val in enumerate(lst)}


def mapping_dict_to_list(dct):

    int_list = [(int(k, 2), int(v, 2)) for k, v in dct.items()]
    return [t[1] for t in sorted(int_list, key=lambda x: x[0])]


def bit_masking(bit_string, mask_string):
    assert len(bit_string) == len(mask_string)
    n_bits = len(bit_string)
    return PADDED_BINARY_BIT_STRING.format(int(bit_string, 2) ^ int(mask_string, 2), n_bits)


def bit_string_orthogonality(bs0, bs1):
    assert len(bs0) == len(bs1)
    n_bits = len(bs0)
    return all([int(bs0[i]) * int(bs1[i]) == 0 for i in range(n_bits)])


def binary_back_substitute(W, s):
    """
    Perform back substitution on a binary system of equations.
    Finds the :math:`\\mathbf{x}` such that
    :math:`\\mathbf{\\mathit{W}}\\mathbf{x}=\\mathbf{s}`,
    where all arithmetic is taken bitwise and modulo 2.
    :param 2darray W: A square :math:`n\\times n` matrix of 0s and 1s,
              in row-echelon form
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
                m[row_num] = s[row_num] ^ s[col_num]

    return m[::-1]
