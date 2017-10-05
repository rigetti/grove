"""A module containing useful utility quantum Programs.
"""

from scipy.linalg import sqrtm
import numpy as np
from collections import Sequence

from pyquil.gates import STANDARD_GATES
import pyquil.quil as pq
from pyquil.quilbase import Qubit

STANDARD_GATE_NAMES = list(STANDARD_GATES.keys())


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


def n_qubit_control(control_qubits, target, operation, gate_name):
    """
    Returns a controlled u gate with n-1 control_qubits.
    Useful for constructing oracles.

    Uses a number of gates quadratic in the number of qubits,
    and defines a linear number of new gates. (Roots and adjoints of unitary.)
    See arXiv:quant-ph/9503016 for more information.

    :param list control_qubits: The indices of the qubits to condition the gate on.
    :param int target: The index of the target of the gate.
    :param numpy.ndarray operation: The unitary gate to be controlled, given as a numpy array.
    :param str gate_name: The name of the gate target.
    :return: The controlled gate.
    """
    if (not isinstance(operation, np.ndarray)
        or len(operation.shape) != 2
        or operation.shape[0] != operation.shape[1]):
        raise ValueError("operation must be a square 2D numpy array")
    if not is_valid_qubits(control_qubits) or len(control_qubits) < 0:
        raise ValueError(
            "controls must be a non-empty Sequence of Qubits, or non-negative integers.")
    if not is_valid_qubits([target]):
        raise ValueError("The target must be a Qubit or non-negative integer.")
    if not isinstance(gate_name, str) or len(gate_name) == 0:
        raise ValueError("gate_name must be a non-empty string.")

    def controlled_program_builder(control_qubits, target, target_gate_name,
                                   target_gate, defined_gates=set(STANDARD_GATE_NAMES)):
        zero_projection = np.array([[1, 0], [0, 0]])
        one_projection = np.array([[0, 0], [0, 1]])

        control_true = np.kron(one_projection, target_gate)
        control_false = np.kron(zero_projection, np.eye(2, 2))
        control_root_true = np.kron(one_projection, sqrtm(target_gate))

        controlled_gate = control_true + control_false
        controlled_root_gate = control_root_true + control_false
        assert np.isclose(controlled_gate, np.dot(controlled_root_gate,
                                                  controlled_root_gate)).all()

        sqrt_name = "SQRT" + target_gate_name
        adj_sqrt_name = "ADJ" + sqrt_name

        # Initialize program and populate with gate information
        p = pq.Program()
        if len(control_qubits) == 0:
            p.defgate(target_gate_name, target_gate)
            p.inst((target_gate_name, target))
            return p, set()

        if len(control_qubits) == 1:
            if "C" + target_gate_name not in defined_gates:
                p.defgate("C" + target_gate_name, controlled_gate)
                defined_gates.add("C" + target_gate_name)
            p.inst(("C" + target_gate_name, control_qubits[0], target))

        else:
            for gate_name, gate in ((sqrt_name, controlled_root_gate),
                                    (adj_sqrt_name,
                                     np.conj(controlled_root_gate.T))):
                if "C" + gate_name not in defined_gates:
                    p.defgate("C" + gate_name, gate)
                    defined_gates.add("C" + gate_name)
            p.inst(("C" + sqrt_name, control_qubits[-1], target))
            many_toff, new_defined_gates = controlled_program_builder(
                control_qubits[:-1], control_qubits[-1], 'NOT', np.array([[0, 1], [1, 0]]),
                set(defined_gates))
            p += many_toff
            defined_gates.union(new_defined_gates)

            p.inst(("C" + adj_sqrt_name, control_qubits[-1], target))

            # Don't redefine all of the gates.
            many_toff.defined_gates = []
            p += many_toff
            many_root_toff, new_defined_gates = controlled_program_builder(
                control_qubits[:-1], target, sqrt_name, sqrtm(target_gate),
                set(defined_gates))
            p += many_root_toff
            defined_gates.union(new_defined_gates)

        return p, defined_gates

    p = controlled_program_builder(control_qubits, target, gate_name, operation)[0]
    return p
