"""Module for the Bernstein-Vazirani Algorithm.
Additional information for this algorithm can be found at: http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/04/lecture04.pdf
"""

import numpy as np
import pyquil.quil as pq
from pyquil.gates import *


def oracle_function(vec_a, b, qubits, ancilla):
    """
    Defines an oracle that performs the following unitary transformation:
    |x>|y> -> |x>|f(x) xor y>

    f(x) is given by a*x+b, where * is a bitwise dot product and everything is taken mod 2.
    :param vec_a: a vector of length n containing only ones and zeros. The order is taken to be
                  most to least significant bit.
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


def run(cxn, vec_a, b):
    """
    Runs the Bernstein-Vazirani algorithm.
    :param cxn: the QVM connection to use to run the programs
    :param vec_a: a vector of 0s and 1s, to represent the a vector.
    :param b: a bit, 0 or 1, to represent the b constant
    :return: a tuple that includes:
                - the program's determination of a
                - the program's determination of b
                - the main program used to determine a
                - the oracle used
    """
    # First, create the program to find a
    qubits = range(len(vec_a))
    ancilla = len(vec_a)

    oracle = oracle_function(vec_a, b, qubits, ancilla)
    bv_program = bernstein_vazirani(oracle, qubits, ancilla)

    results = cxn.run_and_measure(bv_program, qubits)
    bv_a = results[0][::-1]

    # Feed through all zeros to get b
    results = cxn.run_and_measure(oracle, [ancilla])
    bv_b = results[0][0]

    return bv_a, bv_b, bv_program, oracle


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

    qvm = api.SyncConnection()
    a, b, bv_program, oracle = run(qvm, vec_a, b)
    bitstring_a = "".join(map(str, a))
    print "-----------------------------------"
    print "The bitstring a is given by: ", bitstring
    print "b is given by: ", b
    print "-----------------------------------"
    if raw_input("Show Program? (y/n): ") == 'y':
        print "----------Quantum Programs Used----------"
        print "Program to find a given by: "
        print bv_program
        print "Program to find b given by: "
        print oracle