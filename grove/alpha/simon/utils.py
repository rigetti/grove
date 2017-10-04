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


def mapping_list_to_dict(lst):
    n_bits = len(lst).bit_length() - 1
    if len(lst) != 2 ** n_bits:
        raise ValueError("mappings must have a length that is a power of two")

    form_string = "{0:0" + str(n_bits) + "b}"
    return {form_string.format(idx): form_string.format(val) for idx, val in enumerate(lst)}


def mapping_dict_to_list(dct):

    int_list = [(int(k, 2), int(v, 2)) for k, v in dct.items()]
    return [t[1] for t in sorted(int_list, key=lambda x: x[0])]


def bit_masking(bit_string, mask_string):
    assert len(bit_string) == len(mask_string)
    return ''.join([str((int(bit_string[i]) + int(mask_string[i])) % 2) for i in range(len(bit_string))])
