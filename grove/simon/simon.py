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
    Implementation of the quantum portion of Simon's Algorithm.
    For given two-to-one function f: {0,1}^n -> {0,1}^n, determine the non-zero mask s such that
    f(x) = f(y) if and only if (x xor y) = s.
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
                          Would be represented as [0, 2, 2, 0].
            Requires mappings to be two-to-one with unique mask s.
    :return: Matrix representing specified unitary transformation.
    :rtype: numpy array
    """
    n = int(np.log2(len(mappings)))
    distinct_outputs = len(set(mappings))
    assert distinct_outputs in {2**(n-1)}, "Function must be two-to-one"

    # Strategy: add an extra qubit by default and force the function to be one-to-one
    output_counts = {x: 0 for x in range(2**n)}

    unitary_funct = np.zeros(shape=(2 ** (n+1), 2 ** (n+1)))

    # Fill in what is known so far
    for j in range(2 ** n):
        i = mappings[j]
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


def is_unitary(matrix):
    """
    :param matrix: a matrix to test unitarity of
    :return: true if and only if matrix is unitary
    :rtype: bool
    """
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))


def most_significant_bit(lst):
    """
    Finds the position of the most significant bit in a list of 1s and 0s, i.e. the first position where a 1 appears, reading left to right.
    :param lst: a list of 0s and 1s with at least one 1
    :return: the first position in lst that a 1 appears
    :rtype: int
    """
    msb = 0
    while lst[msb] != 1:
        msb += 1
    return msb


if __name__ == "__main__":
    import pyquil.api as api

    # Read function mappings from user
    n = int(input("How many bits? "))
    assert n > 0, "The number of bits must be positive."
    print "Enter f(x) for the following n-bit inputs:"
    mappings = []
    for i in range(2 ** n):
        val = raw_input(np.binary_repr(i, n) + ': ')
        assert all(map(lambda x: x in {'0', '1'}, val)), "f(x) must return only 0 and 1"
        mappings.append(int(val, 2))

    simon_program = pq.Program()
    qubits = range(n)
    ancillas = range(n, 2*n)
    scratch_bit = 2*n

    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancillas, scratch_bit)
    simon_program += simon(oracle, qubits)

    qvm = api.SyncConnection()

    # Generate n-1 linearly independent vectors that will be orthonormal to the mask s
    # Done so by running the Simon program repeatedly and building up a row-echelon matrix W
    # See http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
    iterations = 0
    W = None
    while True:
        if W is not None and len(W) == n-1:
            break
        z = np.array(qvm.run_and_measure(simon_program, [q for q in qubits])[0])
        iterations += 1
        # attempt to insert into W so that W remains in row-echelon form and all rows are linearly independent
        while np.any(z):  # while it's not all zeros
            if W is None:
                W = z
                W = W.reshape(1, n)
                break
            msb_z = most_significant_bit(z)

            # Search for a row to insert z into, so that it has an early significant bit than the row below
            # and a later one than the row above (when reading left-to-right)
            got_to_end = True
            for row_num in range(len((W))):
                row = W[row_num]
                msb_row = most_significant_bit(row)
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
    # can do by adding a vector with a single 1
    # that can be inserted so that diag(W) is all ones
    insert_row_num = 0
    while insert_row_num < n - 1 and W[insert_row_num][insert_row_num] == 1:
        insert_row_num += 1

    new_row = np.zeros(shape=(n,))
    new_row[insert_row_num] = 1
    W = np.insert(W, insert_row_num, new_row, 0)

    s = np.zeros(shape=(n,))
    s[insert_row_num] = 1

    # iterate backwards, starting from second to last row for back-substitution
    for row_num in range(n-2, -1, -1):
        row = W[row_num]
        for col_num in range(row_num+1, n):
            if row[col_num] == 1:
                s[row_num] = int(s[row_num]) ^ int(s[col_num])

    print "The mask s is ", ''.join([str(int(bit)) for bit in s])
    print "Iterations of the algorithm: ", iterations

    if (raw_input("Show Program? (y/n): ") == 'y'):
        print "----------Quantum Program Used----------"
        print simon_program
