"""Module containing implementations of oracle Programs.
"""

import numpy as np

import pyquil.quil as pq
from pyquil.gates import X

import grove.alpha.amplification.amplification as amp


def basis_selector_oracle(bitstring, qubits):
    """
    Defines an oracle that selects the ith element of the computational basis.

    Defines a phase flip rather than bit flip oracle to eliminate need
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