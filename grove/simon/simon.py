"""Module for the Simon's Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np

def oracle_function(unitary_funct, qubits, ancillas, scratch_bit):
    """
    Defines an oracle that performs the following unitary transformation:
    |x>|y> -> |x>|f(x) xor y>
    :param unitary_funct: Matrix representation of the function f, i.e. the
                          unitary transformation that must be applied to a
                          state |x> to get |f(x)>
    :param qubits: List of qubits that enter as input |x>.
    :param ancillas: List of qubits to serve as the ancilliary input |y>.
    :param scratch_bit: Empty qubit to be used as scratch space.
    :return: A program that performs the above unitary transformation.
    :rtype: Program
    """
    assert is_unitary(unitary_funct), "Function must be unitary."
    bits_for_funct = [scratch_bit] + qubits
    p = pq.Program()

    p.defgate("FUNCT", unitary_funct)
    p.defgate("FUNCT-INV", np.linalg.inv(unitary_funct))
    p.inst(tuple(['FUNCT'] + bits_for_funct))
    p.inst(map(lambda qs: CNOT(qs[0], qs[1]), zip(qubits, ancillas)))
    p.inst(tuple(['FUNCT-INV'] + bits_for_funct))
    return p


def simon(oracle, qubits):
    """
    Implementation of Simon's Algorithm.
    For given f: {0,1}^n -> {0,1}^n, determine if f is one-to-one, or two-to-one with a non-zero mask s.
    :param oracle: Program representing unitary application of function.
    :param qubits: List of qubits that enter as state |x>.
    :return: A program corresponding to the desired instance of
             Simon's Algorithm.
    :rtype: Program
    """
    p = pq.Program()

    # Apply Hadamard, Unitary function, and Hadamard again
    p.inst(map(H, qubits))
    p += oracle
    p.inst(map(H, qubits))
    return p


def unitary_function(mappings):
    """
    Creates a unitary transformation that maps each state to the values specified
    in mappings.
    Some (but not all) of these transformations involve a scratch qubit, so one is
    always provided. That is, if given the mapping of n qubits, the calculated transformation
    will be on n + 1 qubits, where the 0th is the scratch bit and the return value
    of the function is left in the qubits that follow.
    :param list mappings: List of the mappings of f(x) on all length n bitstrings.
                          For example, the following mapping:
                          00 -> 00
                          01 -> 10
                          10 -> 10
                          11 -> 00
                          Would be represented as ['00', '10', '10', '00'].
            Requires mappings to either be one-to-one, or two-to-one with unique mask s.
    :return: Matrix representing specified unitary transformation.
    :rtype: numpy array
    """
    n = int(np.log2(len(mappings)))
    distinct_outputs = len(set(mappings))
    assert distinct_outputs in {2**n, 2**(n-1)}, "Function must be one-to-one or two-to-one"

    # Strategy: add an extra qubit by default and force the function to be one-to-one
    output_counts = {x: 0 for x in range(2**n)}

    unitary_funct = np.zeros(shape=(2 ** (n+1), 2 ** (n+1)))

    # Fill in what is known so far
    for j in range(2 ** n):
        i = bitstring_to_integer(mappings[j])
        output_counts[i] += 1
        if output_counts[i] == 2:
            del output_counts[i]
            i += 2 ** n
        unitary_funct[i, j] = 1

    # if one to one, just ignore the scratch bit as it's already unitary
    if distinct_outputs == 2**n:
        return np.kron(np.identity(2), unitary_funct[0:2**n, 0:2**n])

    # otherwise, if two-to-one, fill the array to make it unitary
    # assuming scratch bit will properly be 0
    lower_index = 2 ** n

    for i in output_counts:
        unitary_funct[i, lower_index] = 1
        unitary_funct[i + 2**n, lower_index + 1] = 1
        lower_index += 2

    return unitary_funct

# TODO: merge with Deutsch-Jozsa code and decide on a standard
def integer_to_bitstring(x, n):
    return ''.join([str((x >> i) & 1) for i in range(n-1, -1, -1)])

def bitstring_to_integer(bitstring):
    return reduce(lambda prev, next: prev*2 + next, map(int, bitstring), 0)

def bitstring_to_array(bitstring):
    return np.array(map(int, bitstring))

def is_unitary(matrix):
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

def most_siginifcant_bit(lst):
    msb = 0
    while lst[msb] != 1:
        msb += 1
    return msb

if __name__ == "__main__":
    import pyquil.forest as forest

    # Read function mappings from user
    n = int(input("How many bits? "))
    assert n > 0, "The number of bits must be positive."
    print "Enter f(x) for the following n-bit inputs:"
    mappings = []
    for i in range(2 ** n):
        val = raw_input(integer_to_bitstring(i, n) + ': ')
        assert all(map(lambda x: x in {'0', '1'}, val)), "f(x) must return only 0 and 1"
        mappings.append(val)

    simon_program = pq.Program()
    qubits = [simon_program.alloc() for _ in range(n)]
    ancillas = [simon_program.alloc() for _ in range(n)]
    scratch_bit = simon_program.alloc()

    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancillas, scratch_bit)
    simon_program += simon(oracle, qubits)

    print simon_program
    qvm = forest.Connection()

    # Generate n-1 linearly independent vectors that will be orthonormal to the mask s
    iterations = 0
    W = None
    while True:
        if W is not None and len(W) == n-1:
            break
        z = np.array(qvm.run_and_measure(simon_program, [q.index() for q in qubits])[0])
        iterations += 1
        while np.any(z): # while it's not all zeros
            if W is None:
                W = z
                W = W.reshape(1, n)
                break
            msb_z = most_siginifcant_bit(z)

            got_to_end = True
            for row_num in range(len((W))):
                row = W[row_num]
                msb_row = most_siginifcant_bit(row)
                if msb_row == msb_z:
                    z = np.array([z[i] ^ row[i] for i in range(n)])
                    got_to_end = False
                    break
                elif msb_row > msb_z:
                    W = np.insert(W, row_num, z, 0)
                    got_to_end = False
                    break
            if got_to_end:
                W = np.insert(W, len(W), z, 0)

    # Generate one final vector that is not orthonormal to the mask s
    insert_row_num = 0
    while insert_row_num < n - 1 and W[insert_row_num][insert_row_num] == 1:
        insert_row_num += 1

    new_row = np.zeros(shape=(n,))
    new_row[insert_row_num] = 1
    W = np.insert(W, insert_row_num, new_row, 0)

    s = np.zeros(shape=(n,))
    s[insert_row_num] = 1

    # iterate backwards, starting from second to last row
    for row_num in range(n-2, -1, -1):
        row = W[row_num]
        for col_num in range(row_num+1, n):
            if row[col_num] == 1:
                s[row_num] = int(s[row_num]) ^ int(s[col_num])

    print "The mask s is ", ''.join([str(int(bit)) for bit in s])
    print "Iterations of the algorithm: ", iterations
