"""Module for the Bernstein-Vazirani Algorithm.
Additional information for this algorithm can be found at:
http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/04/lecture04.pdf
"""

import numpy as np
import pyquil.quil as pq
from pyquil.gates import *


def oracle_function(vec_a, b, qubits, ancilla):
    """
    Creates a black box oracle for a function
    to be used in the Bernstein-Vazirani algorithm.

    For a function :math:`f` such that

    .. math::

       f:\\{0,1\\}^n\\rightarrow \\{0,1\\}

       \\mathbf{x}\\rightarrow \\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}

       (\\mathbf{a}\\in\\{0,1\\}^n, b\\in\\{0,1\\})

    where :math:`(\\cdot)` is the bitwise dot product,
    this function defines a program that performs
    the following unitary transformation:

    .. math::

        \\vert \\mathbf{x}\\rangle\\vert y\\rangle \\rightarrow
        \\vert \\mathbf{x}\\rangle
        \\vert f(\\mathbf{x}) \\text{ xor } y\\rangle

    where :math:`\\text{xor}` is taken bitwise.

    Allocates one scratch bit.

    :param list(int) vec_a: a vector of length :math:`n`
                                containing only ones and zeros.
                                The order is taken to be
                                most to least significant bit.
    :param int b: a 0 or 1
    :param list(int) qubits: List of qubits that enter as input
                             :math:`\\vert x\\rangle`.
                             Must be the same length (:math:`n`)
                             as :math:`\\mathbf{a}`
    :param int ancilla: Ancillary qubit to serve as input
                        :math:`\\vert y\\rangle`,
                        where the answer will be written to.
    :return: A program that performs the above unitary transformation.
    :rtype: Program
    """
    assert len(vec_a) == len(qubits), \
        "vec_a must be the same length as the number of input qubits"
    assert all(map(lambda x: x in {0, 1}, vec_a)), \
        "vec_a must be a list of 0s and 1s"
    assert b in {0, 1}, "b must be a 0 or 1"

    n = len(qubits)
    p = pq.Program()
    if b == 1:
        p.inst(X(ancilla))
    for i in xrange(n):
        if vec_a[i] == 1:
            p.inst(CNOT(qubits[n - 1 - i], ancilla))
    return p


def bernstein_vazirani(oracle, qubits, ancilla):
    """
    Implementation of the Bernstein-Vazirani Algorithm.

    Given a list of input qubits and an ancilla bit,
    all initially in the :math:`\\vert 0\\rangle` state,
    create a program that can find :math:`\\vec{a}`
    with one query to the given oracle.

    :param Program oracle: Program representing unitary application of function
    :param list(int) qubits: List of qubits that enter as state
                             :math:`\\vert x\\rangle`.
    :param int ancilla: Ancillary qubit to serve as input
                        :math:`\\vert y\\rangle`,
                        where the answer will be written to.
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


def run_bernstein_vazirani(cxn, oracle, qubits, ancilla):
    """
    Runs the Bernstein-Vazirani algorithm.

    Given a QVM connection, an oracle, the input bits, and ancilla,
    find the :math:`\\mathbf{a}` and :math:`b` corresponding to the function
    represented by the oracle.

    :param Connection cxn: the QVM connection to use to run the programs
    :param Program oracle: the oracle to query that represents
                           a function of the form
                           :math:`f(x)=\\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}`
    :param list qubits: the input qubits
    :param Qubit ancilla: the ancilla qubit
    :return: a tuple that includes, in order,

                * the program's determination of :math:`\\mathbf{a}`
                * the program's determination of :math:`b`
                * the main program used to determine :math:`\\mathbf{a}`
    :rtype: tuple
    """
    # First, create the program to find a
    bv_program = bernstein_vazirani(oracle, qubits, ancilla)

    results = cxn.run_and_measure(bv_program, qubits)
    bv_a = results[0][::-1]

    # Feed through all zeros to get b
    results = cxn.run_and_measure(oracle, [ancilla])
    bv_b = results[0][0]

    return bv_a, bv_b, bv_program


if __name__ == "__main__":
    import pyquil.api as api

    # ask user to input the value for a
    bitstring = raw_input("Give a bitstring representation for the vector a: ")
    while not (all([num in ('0', '1') for num in bitstring])):
        print "The bitstring must be a string of ones and zeros."
        bitstring = raw_input(
            "Give a bitstring representation for the vector a: ")
    vec_a = np.array(map(int, bitstring))

    # ask user to input the value for b
    b = int(raw_input("Give a single bit for b: "))
    while b not in {0, 1}:
        print "b must be either 0 or 1"
        b = int(raw_input("Give a single bit for b: "))

    qvm = api.SyncConnection()
    qubits = range(len(vec_a))
    ancilla = len(vec_a)

    oracle = oracle_function(vec_a, b, qubits, ancilla)

    a, b, bv_program = run_bernstein_vazirani(qvm, oracle, qubits, ancilla)
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
