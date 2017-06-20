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

    :param list vector: A list of complex numbers
    :param offset: Which qubit to begin encoding the state into
    :return: A program that produces the desired state
    :rtype: Program
    """

    # normalize
    norm_vector = vector / np.linalg.norm(vector)

    # pad with zeros
    num_bits = int(np.ceil(np.log2(len(vector))))
    length = int(2 ** num_bits)
    state_vector = np.zeros(length, dtype=complex)
    for i in xrange(len(vector)):
        state_vector[reverse(i, num_bits)] = norm_vector[i]

    # use QR factorization to create unitary
    U = np.identity(length, dtype=complex)
    for i in xrange(length):
        U[i, 0] = state_vector[i]
    Q, R = np.linalg.qr(U)

    # store quantum state as program
    p = pq.Program()
    state_prep_name = "PREP-STATE-{0}".format(hash(p))
    p.defgate(state_prep_name, -1 * Q)
    qubits = [offset + i for i in xrange(num_bits)]
    p.inst(tuple([state_prep_name] + qubits))
    return p

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
