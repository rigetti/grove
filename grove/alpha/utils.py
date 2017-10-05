"""A module containing useful utility quantum Programs.
"""

from scipy.linalg import sqrtm
import numpy as np

from pyquil.gates import STANDARD_GATES
import pyquil.quil as pq

STANDARD_GATE_NAMES = list(STANDARD_GATES.keys())


def n_qubit_control(controls, target, u, gate_name):
    """
    Returns a controlled u gate with n-1 controls.
    Useful for constructing oracles.

    Uses a number of gates quadratic in the number of qubits,
    and defines a linear number of new gates. (Roots and adjoints of u.)

    See arXiv:quant-ph/9503016 for more information.

    :param controls: The indices of the qubits to condition the gate on.
    :param target: The index of the target of the gate.
    :param u: The unitary gate to be controlled, given as a numpy array.
    :param gate_name: The name of the gate u.
    :return: The controlled gate.
    """
    assert isinstance(u, np.ndarray), "The unitary 'u' must be a numpy array"
    assert len(controls) > 0, "The control qubits list must not be empty"
    assert isinstance(target, int) and target > 0, \
        "The target index must be an integer greater than 0"
    assert len(gate_name) > 0, "Gate name must have length greater than one"

    def controlled_program_builder(controls, target, target_gate_name,
                                   target_gate,
                                   defined_gates=set(STANDARD_GATE_NAMES)):
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

        if len(controls) == 1:
            if "C" + target_gate_name not in defined_gates:
                p.defgate("C" + target_gate_name, controlled_gate)
                defined_gates.add("C" + target_gate_name)
            p.inst(("C" + target_gate_name, controls[0], target))

        else:
            for gate_name, gate in ((sqrt_name, controlled_root_gate),
                                    (adj_sqrt_name,
                                     np.conj(controlled_root_gate.T))):
                if "C" + gate_name not in defined_gates:
                    p.defgate("C" + gate_name, gate)
                    defined_gates.add("C" + gate_name)
            p.inst(("C" + sqrt_name, controls[-1], target))
            many_toff, new_defined_gates = controlled_program_builder(
                controls[:-1], controls[-1], 'NOT', np.array([[0, 1], [1, 0]]),
                set(defined_gates))
            p += many_toff
            defined_gates.union(new_defined_gates)

            p.inst(("C" + adj_sqrt_name, controls[-1], target))

            # Don't redefine all of the gates.
            many_toff.defined_gates = []
            p += many_toff
            many_root_toff, new_defined_gates = controlled_program_builder(
                controls[:-1], target, sqrt_name, sqrtm(target_gate),
                set(defined_gates))
            p += many_root_toff
            defined_gates.union(new_defined_gates)

        return p, defined_gates

    p = controlled_program_builder(controls, target, gate_name, u)[0]
    return p