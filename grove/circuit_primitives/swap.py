"""
Implementation of the swap test

Given two states existing on registers A and B, their overlap can be measured by performing a swap
test.
"""
from typing import List

import numpy as np
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import H, CSWAP, MEASURE


class RegisterSizeMismatch(Exception):
    """Error associated with mismatched register size"""
    pass


def swap_circuit_generator(register_a: List[int], register_b: List[int], ancilla: int) -> Program:
    """
    Generate the swap test circuit primitive.

    Registers A and B must be of equivalent size for swap to work.  This module uses the CSWAP gate
    in pyquil.

    :param register_a: qubit labels in the 'A' register
    :param register_b: qubit labels in the 'B' register
    :param ancilla: ancilla to measure and control the swap operation.
    """
    if len(register_a) != len(register_b):
        raise RegisterSizeMismatch("registers involve different numbers of qubits")

    if not isinstance(register_a, list):
        raise TypeError("Register A needs to be list")
    if not isinstance(register_b, list):
        raise TypeError("Register B needs to be a list")

    if ancilla is None:
        ancilla = max(register_a + register_b) + 1

    swap_program = Program()
    swap_program += H(ancilla)
    for a, b in zip(register_a, register_b):
        swap_program += CSWAP(ancilla, a, b)
    swap_program += H(ancilla)

    return swap_program


def run_swap_test(program_a: Program, program_b: Program,
                  number_of_measurements: int,
                  quantum_resource: QuantumComputer,
                  ancilla: int = None) -> float:
    """
    Run a swap test and return the calculated overlap.

    :param program_a: state preparation for the 'A' register
    :param program_b: state preparation for the 'B' register
    :param number_of_measurements: number of times to measure
    :param quantum_resource: connection to QuantumComputer.
    :param ancilla: Index of ancilla
    :return: overlap of states prepared by program_a and program_b
    """
    register_a = list(program_a.get_qubits())
    register_b = list(program_b.get_qubits())
    if ancilla is None:
        ancilla = max(register_a + register_b) + 1

    swap_test_program = Program()
    # instantiate ancilla readout register
    ro = swap_test_program.declare('ro', 'BIT', 1)
    swap_test_program += program_a + program_b
    swap_test_program += swap_circuit_generator(register_a, register_b, ancilla)
    swap_test_program += MEASURE(ancilla, ro[0])

    # TODO: instead of number of measurements have user set precision of overlap
    # estimate and calculate number of shots to be within this precision
    swap_test_program.wrap_in_numshots_loop(number_of_measurements)
    executable = quantum_resource.compiler.native_quil_to_executable(swap_test_program)
    results = quantum_resource.run(executable)

    probability_of_one = np.mean(results)
    if probability_of_one > 0.5:
        raise ValueError("measurements indicate overlap is negative")

    estimated_overlap = np.sqrt(1 - 2 * probability_of_one)
    return estimated_overlap





