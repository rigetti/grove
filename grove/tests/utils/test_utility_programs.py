import numpy as np
from pyquil.quil import Program

from grove.tests.utils.utils_for_testing import non_action_insts, prog_len
from grove.utils.utility_programs import ControlledProgramBuilder

SIGMA_Z = np.array([[1, 0], [0, -1]])
SIGMA_Z_NAME = "Z"


def test_1_qubit_control():
    prog = Program()
    qubit = prog.alloc()
    control_qubit = prog.alloc()
    prog += (ControlledProgramBuilder()
             .with_controls([control_qubit])
             .with_target(qubit)
             .with_operation(SIGMA_Z)
             .with_gate_name(SIGMA_Z_NAME).build())
    # This should be one "CZ" instruction, from control_qubit to qubit.
    assert prog_len(prog) == 1
    prog.synthesize()
    instruction = non_action_insts(prog)[0]
    assert instruction[1].operator_name == (ControlledProgramBuilder()
                                            .format_gate_name("C", SIGMA_Z_NAME))
    assert instruction[1].arguments == [control_qubit, qubit]


def double_control_test(instructions, target_qubit, control_qubit_one, control_qubit_two):
    """A list of asserts testing the simple case of a double controlled Z gate. Used in the next
     two tests."""
    cpg = ControlledProgramBuilder()
    sqrt_z = cpg.format_gate_name("SQRT", SIGMA_Z_NAME)
    assert instructions[0][1].operator_name == (cpg.format_gate_name("C", sqrt_z))
    assert instructions[0][1].arguments == [control_qubit_two, target_qubit]

    assert instructions[1][1].operator_name == cpg.format_gate_name("C", "NOT")
    assert instructions[1][1].arguments == [control_qubit_one, control_qubit_two]

    assert instructions[2][1].operator_name == cpg.format_gate_name("C", sqrt_z) + '-INV'
    assert instructions[2][1].arguments == [control_qubit_two, target_qubit]

    assert instructions[3][1].operator_name == cpg.format_gate_name("C", "NOT")
    assert instructions[3][1].arguments == [control_qubit_one, control_qubit_two]

    assert instructions[4][1].operator_name == cpg.format_gate_name("C", sqrt_z)
    assert instructions[4][1].arguments == [control_qubit_one, target_qubit]


def test_recursive_builder():
    """Here we test the _recursive_builder in ControlledProgramBuilder individually."""
    control_qubit_one = 1
    control_qubit_two = 2
    target_qubit = 3
    cpg = (ControlledProgramBuilder()
           .with_controls([control_qubit_one, control_qubit_two])
           .with_target(target_qubit)
           .with_operation(SIGMA_Z)
           .with_gate_name(SIGMA_Z_NAME))
    prog = cpg._recursive_builder(cpg.operation,
                                  cpg.gate_name,
                                  cpg.control_qubits,
                                  cpg.target_qubit)
    instructions = non_action_insts(prog)
    # Run tests
    double_control_test(instructions, target_qubit, control_qubit_one, control_qubit_two)


def test_2_qubit_control():
    """Test that ControlledProgramBuilder builds the program correctly all the way through."""
    prog = Program()
    qubit = prog.alloc()
    control_qubit_one = prog.alloc()
    control_qubit_two = prog.alloc()
    prog += (ControlledProgramBuilder()
             .with_controls([control_qubit_one, control_qubit_two])
             .with_target(qubit)
             .with_operation(SIGMA_Z)
             .with_gate_name(SIGMA_Z_NAME).build())
    # This should be one "CZ" instruction, from control_qubit to qubit.
    assert prog_len(prog) == 5
    prog.synthesize()
    instructions = non_action_insts(prog)
    # Run tests
    double_control_test(instructions, qubit, control_qubit_one, control_qubit_two)
