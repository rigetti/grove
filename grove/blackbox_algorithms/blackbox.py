"""Module for the Bernstein-Vazirani Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np

def oracle_function(unitary_funct, qubits, ancilla, scratch_bits):
    """
    Defines an oracle that performs the following unitary transformation:
    |x>|y> -> |x>|f(x) xor y>
    :param unitary_funct: Matrix representation of the function f, i.e. the
                          unitary transformation that must be applied to a
                          state |x> to put f(x) in qubit 0.
    :param qubits: List of qubits that enter as input |x>.
    :param ancilla: Qubit to serve as input |y>.
    :param scratch_bits: Empty qubit to be used as scratch space.
    :return: A program that performs the above unitary transformation.
    :rtype: Program
    """
    assert is_unitary(unitary_funct), "Function must be unitary."
    bits_for_funct = scratch_bits + qubits
    p = pq.Program()

    p.defgate("FUNCT", unitary_funct)
    p.defgate("FUNCT-INV", np.linalg.inv(unitary_funct))
    p.inst(tuple(['FUNCT'] + bits_for_funct))
    p.inst(CNOT(qubits[-1], ancilla))
    p.inst(tuple(['FUNCT-INV'] + bits_for_funct))
    return p


def bernstein_vazirani(oracle, qubits, ancilla):
    """
    Implementation of the Bernstein-Vazirani Algorithm.
    For given a in {0,1}^n and b in {0,1}, can determine a with one query to an oracle
    that provides f(x) = a*x+b (mod 2) for x in {0,1}^n.
    :param oracle: Program representing unitary application of function.
    :param qubits: List of qubits that enter as state |x>.
    :param ancilla: Qubit to serve as input |y>.
    :return: A program corresponding to the desired instance of the
             Bernstein-Vazirani Algorithm.
    :rtype: Program
    """
    p = pq.Program()

    # Put ancilla bit into minus state
    p.inst(X(ancilla), H(ancilla))

    # Apply Hadamard, Unitary function, and Hadamard again
    p.inst(map(H, qubits))
    p += oracle
    p.inst(map(H, qubits))
    return p

def integer_to_bitstring(x, n):
    return ''.join([str((x >> i) & 1) for i in range(n)])

def bitstring_to_array(bitstring):
    return np.array(map(int, bitstring))

def is_unitary(matrix):
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

##################
# New stuff

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

def bv_example(vec_a, b):
    def func(x):
        return (int(np.dot(vec_a, bitstring_to_array(integer_to_bitstring(x, len(vec_a))))) + b) % 2

    return func

if __name__ == "__main__":
    import pyquil.forest as forest
    import sys

    if len(sys.argv) != 3:
        raise ValueError("Use program as: python bernstein_vazirani.py vec_a b")
    bitstring = sys.argv[1]
    if not (all([num in ('0', '1') for num in bitstring])):
        raise ValueError("The bitstring must be a string of ones and zeros.")
    if not sys.argv[2] in {'0', '1'}:
        raise ValueError("b must be 0 or 1.")

    vec_a = bitstring_to_array(bitstring)
    b = int(sys.argv[2])

    bv_program = pq.Program()
    qubits = [bv_program.alloc() for _ in range(len(vec_a))]
    ancilla = bv_program.alloc()

    unitary_funct, num_scratch_bits = unitary_from_function(bv_example(vec_a, b), len(vec_a), 1)

    scratch_bits = [bv_program.alloc() for _ in range(num_scratch_bits)]

    oracle = oracle_function(unitary_funct, qubits, ancilla, scratch_bits)
    bv_program += bernstein_vazirani(oracle, qubits, ancilla)

    print bv_program

    qvm = forest.Connection()
    results = qvm.run_and_measure(bv_program, [q.index() for q in qubits])
    print "The bitstring a is given by: ", "".join(map(str, results[0])[::-1])
