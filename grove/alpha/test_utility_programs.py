import numpy as np
import pytest

from pyquil.quil import Program

from grove.alpha.utility_programs import n_qubit_control
from grove.pyquil_utilities import non_action_insts


def test_1_qubit_control():
    prog = Program()
    qubit = prog.alloc()
    control_qubit = prog.alloc()
    sigma_z = np.array([[1, 0], [0, -1]])
    prog += n_qubit_control([control_qubit], qubit, sigma_z, "Z")
    # This should be one "CZ" instruction, from control_qubit to qubit.
    assert len(prog) == 1
    prog.synthesize()
    instruction = non_action_insts(prog)[0]
    assert instruction[1].operator_name == 'CZ'
    assert instruction[1].arguments == [control_qubit, qubit]


def test_2_qubit_control():
    prog = Program()
    qubit = prog.alloc()
    control_qubit_one = prog.alloc()
    control_qubit_two = prog.alloc()
    sigma_z = np.array([[1, 0], [0, -1]])
    prog += n_qubit_control([control_qubit_one, control_qubit_two], qubit, sigma_z, "Z")

    # This should be one "CZ" instruction, from control_qubit to qubit.
    # The circuit should
    assert len(prog) == 5
    prog.synthesize()
    instructions = non_action_insts(prog)
    assert instructions[0][1].operator_name == 'CSQRTZ'
    assert instructions[0][1].arguments == [control_qubit_two, qubit]

    assert instructions[1][1].operator_name == 'CNOT'
    assert instructions[1][1].arguments == [control_qubit_one, control_qubit_two]

    assert instructions[2][1].operator_name == 'CADJSQRTZ'
    assert instructions[2][1].arguments == [control_qubit_two, qubit]

    assert instructions[3][1].operator_name == 'CNOT'
    assert instructions[3][1].arguments == [control_qubit_one, control_qubit_two]

    assert instructions[4][1].operator_name == 'CSQRTZ'
    assert instructions[4][1].arguments == [control_qubit_one, qubit]


def test_bad_n_control():
    prog = Program()
    qubit = prog.alloc()
    control_qubit = prog.alloc()
    control_qubits = [control_qubit]
    sigma_z = np.array([[1, 0], [0, -1]])
    with pytest.raises(ValueError):
        not_array = [1, 0]
        n_qubit_control(control_qubits, qubit, not_array, "Z")
    with pytest.raises(ValueError):
        not_2d_array = np.array([1, 0])
        n_qubit_control(control_qubits, qubit, not_2d_array, "Z")
    with pytest.raises(ValueError):
        not_square_array = np.array([[1, 0, 0], [0, 0, 1]])
        n_qubit_control(control_qubits, qubit, not_square_array, "Z")
    with pytest.raises(ValueError):
        bad_qubits = ["foo"]
        n_qubit_control(bad_qubits, qubit, sigma_z, "Z")
    with pytest.raises(ValueError):
        bad_qubit = "foo"
        n_qubit_control(control_qubits, bad_qubit, sigma_z, "Z")
    with pytest.raises(ValueError):
        not_a_string = []
        n_qubit_control(control_qubits, qubit, sigma_z, not_a_string)
