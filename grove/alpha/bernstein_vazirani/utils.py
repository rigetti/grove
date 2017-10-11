from operator import xor

PADDED_BINARY_BIT_STRING = "{0:0{1:0d}b}"


def bitwise_dot_product(bs0, bs1):
    """
    A helper to calculate the bitwise dot-product between two string representing bit-vectors

    :param String bs0: String of 0's and 1's representing a number in binary representations
    :param String bs1: String of 0's and 1's representing a number in binary representations
    :return: 0 or 1 as a string corresponding to the dot-product value
    :rtype: String
    """
    if len(bs0) != len(bs1):
        raise ValueError("Bit strings are not of equal length")
    return str(sum([int(bs0[i]) * int(bs1[i]) for i in range(len(bs0))]) % 2)


def bitwise_xor(bs0, bs1):
    """
    A helper to calculate the bitwise XOR of two bit string

    :param String bs0: String of 0's and 1's representing a number in binary representations
    :param String bs1: String of 0's and 1's representing a number in binary representations
    :return: String of 0's and 1's representing the XOR between bs0 and bs1
    :rtype: String
    """
    if len(bs0) != len(bs1):
        raise ValueError("Bit strings are not of equal length")
    n_bits = len(bs0)
    return PADDED_BINARY_BIT_STRING.format(xor(int(bs0, 2), int(bs1, 2)), n_bits)
