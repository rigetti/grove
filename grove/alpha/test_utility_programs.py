import numpy as np

from pyquil.quil import Program

from grove.test_utilities import non_action_insts, prog_len
from grove.alpha.utility_programs import ControlledProgramBuilder


def test_1_qubit_control():
    prog = Program()
    qubit = prog.alloc()
    control_qubit = prog.alloc()
    sigma_z = np.array([[1, 0], [0, -1]])
    prog += ControlledProgramBuilder().with_controls([control_qubit]).with_target(
        qubit).with_operation(sigma_z).with_gate_name("Z").build()
    # This should be one "CZ" instruction, from control_qubit to qubit.
    assert prog_len(prog) == 1
    prog.synthesize()
    instruction = non_action_insts(prog)[0]
    assert instruction[1].operator_name == 'C[Z]'
    assert instruction[1].arguments == [control_qubit, qubit]


def test_2_qubit_control():
    prog = Program()
    qubit = prog.alloc()
    control_qubit_one = prog.alloc()
    control_qubit_two = prog.alloc()
    sigma_z = np.array([[1, 0], [0, -1]])
    prog += ControlledProgramBuilder().with_controls(
        [control_qubit_one, control_qubit_two]).with_target(qubit).with_operation(
        sigma_z).with_gate_name("Z").build()

    # This should be one "CZ" instruction, from control_qubit to qubit.
    # The circuit should
    assert prog_len(prog) == 5
    prog.synthesize()
    instructions = non_action_insts(prog)
    assert instructions[0][1].operator_name == 'C[SQRT[Z]]'
    assert instructions[0][1].arguments == [control_qubit_two, qubit]

    assert instructions[1][1].operator_name == 'C[NOT]'
    assert instructions[1][1].arguments == [control_qubit_one, control_qubit_two]

    assert instructions[2][1].operator_name == 'C[SQRT[Z]]-INV'
    assert instructions[2][1].arguments == [control_qubit_two, qubit]

    assert instructions[3][1].operator_name == 'C[NOT]'
    assert instructions[3][1].arguments == [control_qubit_one, control_qubit_two]

    assert instructions[4][1].operator_name == 'C[SQRT[Z]]'
    assert instructions[4][1].arguments == [control_qubit_one, qubit]
