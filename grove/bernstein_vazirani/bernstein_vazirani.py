"""Module for the Bernstein-Vazirani Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np

def oracle_function(unitary_funct, qubits, ancilla, scratch_bit):
    """
    Defines an oracle that performs the following unitary transformation:
    |x>|y> -> |x>|f(x) xor y>
    :param unitary_funct: Matrix representation of the function f, i.e. the
                          unitary transformation that must be applied to a
                          state |x> to put f(x) in qubit 0.
    :param qubits: List of qubits that enter as input |x>.
    :param ancilla: Qubit to serve as input |y>.
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
    p.inst(CNOT(qubits[0], ancilla))
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


def unitary_function(vec_a, b):
    """
    Creates a unitary transformation that maps each state to the value specified by vec_a and b.
    Some (but not all) of these transformations involve a scratch qubit, so one is
    always provided. That is, if given the mapping of n qubits, the calculated transformation
    will be on n + 1 qubits, where the 0th is the scratch bit and the return value
    of the function is left in the 1st.
    :param numpy array vec_a:
    :param int b:
    Matrix representing specified unitary transformation.
    :rtype: numpy array
    """
    n = len(vec_a)
    unitary_funct = np.zeros(shape=(2 ** n, 2 ** n))
    index_lists = [range(2 ** (n - 1)), range(2 ** (n - 1), 2 ** n)]
    if sum(vec_a) == 0:
        SWAP_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                [0, 1, 0, 0], [0, 0, 0, 1]])

        return np.kron(SWAP_matrix, np.identity(2 ** (n - 1)))
    for j in range(2 ** n):
        val = (int(np.dot(vec_a, bitstring_to_array(integer_to_bitstring(j, n)))) + b) % 2
        i = index_lists[val].pop()
        unitary_funct[i, j] = 1
    return np.kron(np.identity(2), unitary_funct)

def integer_to_bitstring(x, n):
    return ''.join([str((x >> i) & 1) for i in range(0, n)])

def bitstring_to_array(bitstring):
    return np.array(map(int, bitstring))

def is_unitary(matrix):
    rows, cols = matrix.shape
    if rows != cols:
        return False
    return np.allclose(np.eye(rows), matrix.dot(matrix.T.conj()))

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
    scratch_bit = bv_program.alloc()

    unitary_funct = unitary_function(vec_a, b)
    oracle = oracle_function(unitary_funct, qubits, ancilla, scratch_bit)
    bv_program += bernstein_vazirani(oracle, qubits, ancilla)
    bv_program.out()

    print bv_program
    qvm = forest.Connection()
    results = qvm.run_and_measure(bv_program, [q.index() for q in qubits])
    print "The bitstring a is given by: ", "".join(map(str, results[0][::-1]))