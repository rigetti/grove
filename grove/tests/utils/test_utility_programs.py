import numpy as np
from pyquil.quil import Program
from pyquil.quilbase import Qubit

from grove.utils.utility_programs import ControlledProgramBuilder
from pyquil.gates import CNOT

SIGMA_Z = np.array([[1, 0], [0, -1]])
SIGMA_Z_NAME = "Z"


def test_1_qubit_control():
    prog = Program()
    qubit = Qubit(0)
    control_qubit = Qubit(1)
    prog += (ControlledProgramBuilder()
             .with_controls([control_qubit])
             .with_target(qubit)
             .with_operation(SIGMA_Z)
             .with_gate_name(SIGMA_Z_NAME).build())
    # This should be one "CZ" instruction, from control_qubit to qubit.
    assert len(prog) == 1
    instruction = prog.instructions[0]
    assert instruction.name == (ControlledProgramBuilder()
                                .format_gate_name("C", SIGMA_Z_NAME))
    assert instruction.qubits == [control_qubit, qubit]


def double_control_test(instructions, target_qubit, control_qubit_one, control_qubit_two):
    """A list of asserts testing the simple case of a double controlled Z gate. Used in the next
     two tests."""
    cpg = ControlledProgramBuilder()
    sqrt_z = cpg.format_gate_name("SQRT", SIGMA_Z_NAME)
    assert instructions[0].name == (cpg.format_gate_name("C", sqrt_z))
    assert instructions[0].qubits == [control_qubit_two, target_qubit]

    assert instructions[1].name == CNOT(control_qubit_one, control_qubit_two).name
    assert instructions[1].qubits == [control_qubit_one, control_qubit_two]

    assert instructions[2].name == cpg.format_gate_name("C", sqrt_z) + '-INV'
    assert instructions[2].qubits == [control_qubit_two, target_qubit]

    assert instructions[3].name == CNOT(control_qubit_one, control_qubit_two).name
    assert instructions[3].qubits == [control_qubit_one, control_qubit_two]

    assert instructions[4].name == cpg.format_gate_name("C", sqrt_z)
    assert instructions[4].qubits == [control_qubit_one, target_qubit]


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
    # Run tests
    double_control_test(prog.instructions,
                        Qubit(target_qubit),
                        Qubit(control_qubit_one),
                        Qubit(control_qubit_two))


def test_2_qubit_control():
    """Test that ControlledProgramBuilder builds the program correctly all the way through."""
    prog = Program()
    qubit = Qubit(0)
    control_qubit_one = Qubit(1)
    control_qubit_two = Qubit(2)
    prog += (ControlledProgramBuilder()
             .with_controls([control_qubit_one, control_qubit_two])
             .with_target(qubit)
             .with_operation(SIGMA_Z)
             .with_gate_name(SIGMA_Z_NAME).build())
    # This should be one "CZ" instruction, from control_qubit to qubit.
    assert len(prog) == 5
    # Run tests
    double_control_test(prog.instructions, qubit, control_qubit_one, control_qubit_two)
