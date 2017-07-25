"""Module for Grover's algorithm.
Based off standalone grover implementation.
Uses the amplitude amplification library."""

import numpy as np
import pyquil.quil as pq
from pyquil.gates import H, X, STANDARD_GATES

import amplification as amp

STANDARD_GATE_NAMES = STANDARD_GATES.keys()


def grover(oracle, qubits, num_iter=None):
    """
    Implementation of Grover's Algorithm for a given oracle.

    The query qubit will be left in the zero state afterwards.

    :param Program oracle: An oracle defined as a Program.
                           It should send |x> to (-1)^f(x)|x>,
                           where the range of f is {0, 1}.
    :param list(int) qubits: List of qubits for Grover's Algorithm.
    :param int num_iter: The number of iterations to repeat the algorithm for.
                         The default is the integer closest to
                         :math:`\\frac{\\pi}{4}\sqrt{N}`, where :math:`N` is
                         the size of the domain.
    :return: A program corresponding to
             the desired instance of Grover's Algorithm.
    :rtype: Program
    """
    if len(qubits) < 1:
        raise ValueError("Grover's Algorithm requires at least 1 qubits.")

    if num_iter is None:
        num_iter = int(round(np.pi * 2 ** (len(qubits) / 2.0 - 2.0)))

    many_hadamard = pq.Program().inst(map(H, qubits))
    amp_prog = amp.amplify(many_hadamard, many_hadamard,
                           oracle, qubits, num_iter)

    return amp_prog


def basis_selector_oracle(bitstring, qubits):
    """
    Defines an oracle that selects the ith element of the computational basis.

    Defines a phase filp rather than bit flip oracle to eliminate need
    for extra qubit. Flips the sign of the state :math:`\\vert x\\rangle>`
    if and only if x==bitstring and does nothing otherwise.

    :param bitstring: The desired bitstring,
                      given as a string of ones and zeros. e.g. "101"
    :param qubits: The qubits the oracle is called on.
                   The qubits are assumed to be ordered from most
                   significant qubit to least significant qubit.
    :return: A program representing this oracle.
    """
    if len(qubits) != len(bitstring):
        raise ValueError(
            "The bitstring should be the same length as the number of qubits.")
    if not (isinstance(bitstring, str) and all(
            [num in ('0', '1') for num in bitstring])):
        raise ValueError("The bitstring must be a string of ones and zeros.")
    prog = pq.Program()
    for i, qubit in enumerate(qubits):
        if bitstring[i] == '0':
            prog.inst(X(qubit))

    prog += amp.n_qubit_control(qubits[:-1], qubits[-1],
                                np.array([[1, 0], [0, -1]]), 'Z')

    for i, qubit in enumerate(qubits):
        if bitstring[i] == '0':
            prog.inst(X(qubit))
    return prog


if __name__ == "__main__":
    from pyquil.api import SyncConnection
    import sys

    try:
        target = sys.argv[1]
    except IndexError:
        raise ValueError("Enter a target bitstring for Grover's Algorithm.")

    grover_program = pq.Program()
    qubits = range(len(target))
    oracle = basis_selector_oracle(target, qubits)
    grover_program += grover(oracle, qubits)

    cxn = SyncConnection()
    mem = cxn.run_and_measure(grover_program, qubits)
    print mem
