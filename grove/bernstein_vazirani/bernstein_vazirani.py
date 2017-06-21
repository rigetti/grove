"""Module for the Bernstein-Vazirani Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np

def oracle_function(vec_a, b, qubits, ancilla):
    """
    Defines an oracle that performs the following unitary transformation:
    |x>|y> -> |x>|f(x) xor y>

    f(x) is given by a*x+b, where * is a bitwise dot product and everything is taken mod 2.
    :param vec_a: a vector of length n containing only ones and zeros.
    :param b: a 0 or 1
    :param qubits: List of qubits that enter as input |x>. Must be the same length (n) as vec_a
    :param ancilla: Qubit to serve as input |y>, where the answer will be written to.
    :return: A program that performs the above unitary transformation.
    :rtype: Program
    """
    n = len(qubits)
    p = pq.Program()
    if b == 1:
        p.inst(X(ancilla))
    for i in xrange(n):
        if vec_a[i] == 1:
            p.inst(CNOT(qubits[n-1-i], ancilla))
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

if __name__ == "__main__":
    import pyquil.forest as forest

    # ask user to input the value for a
    bitstring = raw_input("Give a bitstring representation for the vector a: ")
    while not (all([num in ('0', '1') for num in bitstring])):
        print "The bitstring must be a string of ones and zeros."
        bitstring = raw_input("Give a bitstring representation for the vector a: ")
    vec_a = np.array(map(int, bitstring))

    # ask user to input the value for b
    b = int(raw_input("Give a single bit for b: "))
    while b not in {0, 1}:
        print "b must be either 0 or 1"
        b = int(raw_input("Give a single bit for b: "))


    # First, create the program to find a
    bv_program = pq.Program()
    qubits = [bv_program.alloc() for _ in range(len(vec_a))]
    ancilla = bv_program.alloc()

    oracle = oracle_function(vec_a, b, qubits, ancilla)
    bv_program += bernstein_vazirani(oracle, qubits, ancilla)
    bv_program.out()

    qvm = forest.Connection()
    results = qvm.run_and_measure(bv_program, [q.index() for q in qubits])
    print "The bitstring a is given by: ", "".join(map(str, results[0][::-1]))

    # Reset the qubits to all 0s and feed through to get b
    bv_program.inst(RESET)
    bv_program += oracle
    results = qvm.run_and_measure(bv_program, [ancilla.index()])

    print "b is given by: ", results[0][0]

    print "Full program given by: "
    print bv_program
