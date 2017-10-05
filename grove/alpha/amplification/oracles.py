"""Module containing implementations of oracle Programs.
"""

import numpy as np
from collections import Sequence

import pyquil.quil as pq
from pyquil.gates import X
from pyquil.quilbase import Qubit

import grove.alpha.amplification.amplification as amp


def basis_selector_oracle(qubits, bitstring):
    """Defines an oracle that selects the ith element of the computational basis.

    Defines a phase flip rather than bit flip oracle to eliminate need
    for extra qubit. Flips the sign of the state :math:`\\vert x\\rangle>`
    if and only if x==bitstring and does nothing otherwise.

    :param qubits: The qubits the oracle is called on. The qubits are assumed to be ordered from
     most significant qubit to least significant qubit.
    :param bitstring: The desired bitstring,
     given as a string of ones and zeros. e.g. "101"
    :return: A program representing this oracle.
    :rtype: Program
    """
    if not (isinstance(bitstring, str) and all([num in ('0', '1') for num in bitstring])):
        raise ValueError("bitstring must be a string of ones and zeros.")
    if not (isinstance(qubits, Sequence)
            and all([isinstance(qubit, (Qubit, int)) for qubit in qubits])):
        raise ValueError("qubits must be a list of integers and/or Qubits.")
    if len(qubits) != len(bitstring):
        raise ValueError(
            "The bitstring should be the same length as the number of qubits.")

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