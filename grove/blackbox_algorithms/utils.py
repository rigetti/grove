"""Utils for Black Box Algorithm."""

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

def unitary_from_function(func, num_domain_bits, num_range_bits):
    """
    Creates a unitary function from a given function, assuming that the range space is not larger than the domain space,
    i.e. num_domain_bits >= num_range_bits.

    This makes sense given that blackbox problems typically map the domain down to a space of smaller or equal size

    :param func: function that maps {0,1}^num_domain_bits to {0,1}^num_range_bits
                 function should be callable as f(x), where x is in [0, 2^num_domain_bits) and f(x) is in [0, 2^num_range_bits)
    :param num_domain_bits: the number such that {0,1}^num_domain_bits is the domain of the function
    :param num_range_bits: the number such that {0,1}^num_range_bits is the domain of the function
    :return: A tuple (U, num_scratch_bits) that gives the unitary U, and the number of scratch bits num_scratch_bits
             that must be supplied as |0>s.

             The application of U on |s>|x>, where |s> is num_scratch_bits scratch bits then returns
             |f(x)> on top of the first num_range_bits of where |x> originally was (least significant bits)
             There are no other guarantees on the state of the rest of the bits, or the scratch bits.
    """
    counts = {y: 0 for y in range(2**num_range_bits)}
    max_count = 0
    for x in range(2**num_domain_bits):
        y = func(x)
        counts[y] += 1
        if counts[y] > max_count:
            max_count = counts[y]

    # Strategy: add extra qubits as needed and force the function to be one-to-one
    # We need log_2(max_count) scratch bits, of which we can use num_domain_bits - num_range_bits for
    new_bits = max(0, (int(np.ceil(np.log2(max_count)))) - (num_domain_bits - num_range_bits))

    U = np.zeros(shape=(2 ** (num_domain_bits + new_bits), 2 ** (num_domain_bits + new_bits)))

    # keep track of how many range outputs have been placed, and which rows/columns are still available
    # unitarity is enforced by having exactly one 1 in each row and column
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

    # fill 1s into rows/columns that are missing them
    for i, j in zip(list(rows_free), list(columns_free)):
        U[i, j] = 1

    return (U, new_bits)

def integer_to_bitstring(x, n):
    """
    :param x: a positive base 10 integer
    :param n: the number of bits that the bitstring should be
    :return: the lowest n significant digits of the binary representation of x
             with 0s padded if needed. Significance decreases from left to right.
    """
    return ''.join([str((x >> i) & 1) for i in range(n-1, -1, -1)])

def bitstring_to_array(bitstring):
    """
    :param bitstring: bitstring to convert into an array
    :return: the array corresponding to the bitstring
    """
    return np.array(map(int, bitstring))

def bitstring_to_integer(bitstring):
    """
    :param bitstring: The binary string to convert
    :return: the base 10 number corresponding bitstring, presumed to be in binary.
    """
    return reduce(lambda prev, next: prev*2 + next, map(int, bitstring), 0)

def is_unitary(matrix):
    """
    :param matrix: a matrix to test unitarity of
    :return: true if and only if matrix is unitary
    """
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

def get_n_bits(prog, n):
    """
    Produces n new qubits for a program and returns them in a list
    :param prog: the program from which to allocate qubits
    :param n: the number of qubits to allocate
    :return: a list of the n allocated qubits
    """
    return [prog.alloc() for _ in range(n)]

def most_siginifcant_bit(lst):
    """
    Finds the position of the most significant bit in a list of 1s and 0s, i.e. the first position where a 1 appears, reading left to right.
    :param lst: a list of 0s and 1s
    :return: the first position in lst that a 1 appears
    """
    msb = 0
    while lst[msb] != 1:
        msb += 1
    return msb
