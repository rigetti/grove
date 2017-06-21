"""Module for the Bernstein-Vazirani Algorithm.
Additional information for this algorithm can be found at: http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/04/lecture04.pdf
"""

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
    import pyquil.api as api

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

    print "-----------------------------------"
    # First, create the program to find a
    bv_program = pq.Program()
    qubits = [bv_program.alloc() for _ in range(len(vec_a))]
    ancilla = bv_program.alloc()

    oracle = oracle_function(vec_a, b, qubits, ancilla)
    bv_program += bernstein_vazirani(oracle, qubits, ancilla)
    bv_program.out()

    qvm = api.SyncConnection()
    results = qvm.run_and_measure(bv_program, [q.index() for q in qubits])
    print "The bitstring a is given by: ", "".join(map(str, results[0][::-1]))

    # Feed through all zeros to get b
    results = qvm.run_and_measure(oracle, [ancilla.index()])

    print "b is given by: ", results[0][0]
    print "-----------------------------------"
    print "Full program given by: "
    print bv_program
