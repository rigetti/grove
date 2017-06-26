"""
Module for encoding any complex vector into the wavefunction of a
quantum state. For example, the input vector [a, b, c, d] would result
in the state a|00> + b|01> + c|10> + d|11>.
"""
import pyquil.quil as pq
from pyquil.api import SyncConnection
import numpy as np

def generate_state(vector, offset=0):
    """
    Encodes an arbitrary vector into a quantum state.
    The input vector is normalized and padded with zeros, if necessary.

    :param 1d array vector: A list of complex numbers
    :param offset: Which qubit to begin encoding the state into
    :return: A program that produces the desired state
    :rtype: Program
    """

    num_bits = int(np.ceil(np.log2(len(vector))))

    # normalize
    norm_vector = vector / np.linalg.norm(vector)

    # pad with zeros
    length = next_power_of_two(len(vector))
    state_vector = np.zeros(length, dtype=complex)
    for i in range(len(vector)):
        state_vector[reverse(i, num_bits)] = norm_vector[i]

    # use QR factorization to create unitary
    U = unitary_operator(state_vector)

    # store quantum state as program
    p = pq.Program()
    state_prep_name = "PREP-STATE-{0}".format(hash(p))
    p.defgate(state_prep_name, U)
    qubits = [offset + i for i in range(num_bits)]
    p.inst(tuple([state_prep_name] + qubits))
    return p

def unitary_operator(state_vector):
    """
    Uses QR factorization to create a unitary operator that encodes an arbitrary normalized
    vector into the wavefunction of a quantum state.

    :param 1d array state_vector: Normalized vector whose length is power of two
    :return: Unitary operator that encodes state_vector
    :rtype: 2d array
    """

    assert np.allclose([np.linalg.norm(state_vector)], [1]), \
        "Vector must be normalized"
    assert next_power_of_two(len(state_vector)) == len(state_vector), \
        "Vector length must be a power of two"

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
        return -1*U

def next_power_of_two(n):
    """
    Calculates the first power of two >= n.

    :param int n: A positive integer
    :return: The power of two that n rounds up to
    :rtype: int
    """

    assert n > 0, "Inout should be positive"
    num_bits = int(np.ceil(np.log2(n)))
    return int(2 ** num_bits)

def reverse(x, n):
    """
    Reverses the significance of an integers bits in binary.

    :param int x: The integer to be reversed
    :param int n: The number of bits with which to represent x
    :return: The n-bit representation of x with the significance
             of the bits reversed
    :rtype: int
    """

    result = 0
    for i in xrange(n):
        if (x >> i) & 1: result |= 1 << (n - 1 - i)
    return result

if __name__ == "__main__":
    num_entries = int(raw_input("How long is the input vector? "))
    num_bits = int(np.ceil(np.log2(num_entries)))
    print "Begin entering vector elements below."
    vector = []
    for i in xrange(num_entries):
        vector.append(complex(raw_input("Element {0}: ".format(i))))
    print "You entered the following vector:", vector
    print "The following Quil code was generated:"
    p = generate_state(vector)
    print p
    print "The vector has been normalized and encoded into the following {0}-qubit state:".format(num_bits)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)
    print wf
