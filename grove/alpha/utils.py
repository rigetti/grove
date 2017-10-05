from collections import Sequence

from pyquil.quilbase import Qubit


def is_valid_qubits(qubits):
    """Checks that qubits is a valid list of qubit-like objects.

    :param Sequence qubits: Sequence of qubit-like objects (ints or Qubits).
    :return: True if qubits is a valid Sequence of qubits, False otherwise.
    :rtype: bool
    """
    if isinstance(qubits, Sequence):
        for qubit in qubits:
            if not isinstance(qubit, (Qubit, int)):
                return False
            if isinstance(qubit, int):
                if qubit < 0:
                    return False
        return True
    return False
