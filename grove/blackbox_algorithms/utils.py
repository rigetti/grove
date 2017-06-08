"""Utils for Black Box Algorithms"""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np

def oracle_function(unitary_funct, qubits, ancillas, scratch_bits):
    """
    Defines an oracle that performs the following unitary transformation:
    |x>|y> -> |x>|f(x) xor y>
    where xor is computed bitwise
    :param unitary_funct: Matrix representation of the function f, i.e. the
                          unitary transformation that must be applied to a
                          state |x> to put f(x) in qubit 0.
    :param qubits: List of qubits that enter as input |x>.
    :param ancillas: Qubits to serve as the ancilliary input |y>.
    :param scratch_bits: Empty |0> qubits to be used as scratch space.
    :return: A program that performs the above unitary transformation.
    :rtype: Program
    """
    assert is_unitary(unitary_funct), "Function must be unitary."
    bits_for_funct = scratch_bits + qubits  # scratch bits are placed in the highest significant positions
    p = pq.Program()

    p.defgate("FUNCT", unitary_funct)
    p.defgate("FUNCT-INV", unitary_funct.T.conj())  # inverse of a unitary is given by the conjugate transpose

    p.inst(tuple(['FUNCT'] + bits_for_funct))
    p.inst(map(lambda qs: CNOT(qs[0], qs[1]), zip(qubits[-len(ancillas):], ancillas)))  # copy bits from lowest bits onto ancillas
    p.inst(tuple(['FUNCT-INV'] + bits_for_funct))

    return p

# TODO: for now the assumption is num_domain_bits >= num_range_bits, which makes sense given that problems typically map the domain down to a smaller space
# TODO: actually understand the right-to-left standard of qubits, may mean making some changes to the code to better support this


def unitary_from_function(func, num_domain_bits, num_range_bits):
    """
    :param func: function that maps {0,1}^num_domain_bits to {0,1}^num_range_bits
                 function should be callable as f(x), where x is in [0, 2^num_domain_bits) and f(x) is in [0, 2^num_range_bits)
    :param num_domain_bits: the number such that {0,1}^num_domain_bits is the domain of the function
    :param num_range_bits: the number such that {0,1}^num_range_bits is the domain of the function
    :return: A tuple (U, num_scratch_bits) that gives the unitary U, and the number of scratch bits num_scratch_bits
             that must be supplied as |0>s.

             The application of U on |s>|x>, where |s> is num_scratch_bits scratch bits then returns
             |f(x)> on top of the first num_range_bits of where |x> originally was
             There are no other guarantees on the state of the rest of the bits, or the scratch bits.
    """
    counts = {y: 0 for y in range(2**num_range_bits)}
    max_count = 0
    for x in range(2**num_domain_bits):
        y = func(x)
        counts[y] += 1
        if counts[y] > max_count:
            max_count = counts[y]

    # We need log_2(max_count) scratch bits, of which we can use num_domain_
    new_bits = max(0, (int(np.ceil(np.log2(max_count)))) - (num_domain_bits - num_range_bits))

    U = np.zeros(shape=(2 ** (num_domain_bits + new_bits), 2 ** (num_domain_bits + new_bits)))

    # Strategy: add an extra qubit by default and force the function to be one-to-one
    output_counts = {y: 0 for y in range(2**num_range_bits)}
    rows_free = set(range(2 ** (num_domain_bits + new_bits)))
    columns_free = set(range(2 ** (num_domain_bits + new_bits)))

    # Fill in what is known so far
    for j in range(2 ** num_domain_bits):
        y = func(j)
        i = y + ((2 ** num_range_bits) * output_counts[y])
        output_counts[y] += 1
        if output_counts[y] == counts[y]: # we're done with this range value
            del output_counts[y]

        U[i, j] = 1
        rows_free.remove(i)
        columns_free.remove(j)

    for i, j in zip(list(rows_free), list(columns_free)):
        U[i, j] = 1

    return (U, new_bits)

def integer_to_bitstring(x, n):
    return ''.join([str((x >> i) & 1) for i in range(n-1, -1, -1)])

def bitstring_to_array(bitstring):
    return np.array(map(int, bitstring))

def is_unitary(matrix):
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

def get_n_bits(prog, n):
    return [prog.alloc() for _ in range(n)]