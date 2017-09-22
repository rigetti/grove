"""
Module for creating a unitary operator for encoding any complex vector
into the wavefunction of a quantum state. For example, the input vector
:math:`[a, b, c, d]` would result in the state

.. math::

    a\\vert 00\\rangle + b\\vert01\\rangle
    + c\\vert10\\rangle + d\\vert11\\rangle

"""
import numpy as np
from six.moves import input


def unitary_operator(state_vector):
    """
    Uses QR factorization to create a unitary operator
    that can encode an arbitrary normalized vector into
    the wavefunction of a quantum state.

    Assumes that the state of the input qubits is to be expressed as

    .. math::

        (1, 0, \\ldots, 0)^T

    :param 1d array state_vector: Normalized vector whose length is at least
                                  two and a power of two.
    :return: Unitary operator that encodes state_vector
    :rtype: 2d array
    """

    if not np.allclose([np.linalg.norm(state_vector)], [1]):
        raise ValueError("Vector must be normalized")

    if 2 ** get_bits_needed(len(state_vector)) != len(state_vector):
        raise ValueError("Vector length must be a power of two and at least two")

    mat = np.identity(len(state_vector), dtype=complex)
    for i in range(len(state_vector)):
        mat[i, 0] = state_vector[i]
    U = np.linalg.qr(mat)[0]

    # make sure U|0> = |v>
    zero_state = np.zeros(len(U))
    zero_state[0] = 1
    if np.allclose(U.dot(zero_state), state_vector):
        return U
    else:
        # adjust phase if needed
        return -1 * U


def fix_norm_and_length(vector):
    """
    Create a normalized and zero padded version of vector.

    :param 1darray vector: a vector with at least one nonzero component.
    :return: a vector that is the normalized version of vector,
             padded at the end with the smallest number of 0s necessary
             to make the length of the vector :math:`2^m`
             for some positive integer :math:`m`.
    :rtype: 1darray
    """
    # normalize
    norm_vector = vector / np.linalg.norm(vector)

    # pad with zeros
    num_bits = get_bits_needed(len(vector))
    state_vector = np.zeros(2 ** num_bits, dtype=complex)
    for i in range(len(vector)):
        state_vector[i] = norm_vector[i]

    return state_vector


def get_bits_needed(n):
    """
    Calculates the smallest positive integer :math:`m` for which
    :math:`2^m\geq n`.

    :param int n: A positive integer
    :return: The positive integer :math:`m`, as specified above
    :rtype: int
    """

    assert n > 0, "Inout should be positive"

    num_bits = int(np.ceil(np.log2(n)))
    return max(1, num_bits)


if __name__ == "__main__":
    num_entries = int(input("How long is the input vector? "))
    num_bits = int(np.ceil(np.log2(num_entries)))
    print("Begin entering vector elements below.")
    vector = []
    for i in range(num_entries):
        vector.append(complex(input("Element {0}: ".format(i))))
    print("You entered the following vector: ", vector)
    state_vector = fix_norm_and_length(vector)
    print("The following vector will be encoded: ", state_vector)
    print("The following matrix was generated: ")
    mat = unitary_operator(state_vector)
    print(mat)
    print("The first column of this matrix is: ")
    print(list(mat[:, 0]))
