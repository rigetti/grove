"""
Implementation of the swap test

Given two states existing on registers A and B, their overlap can be measured
by performing a swap test.
"""
import numpy as np
from pyquil.quil import Program
from pyquil.gates import H, CSWAP


class RegisterSizeMismatch(Exception):
    """Error associated with mismatched register size"""
    pass


def swap_circuit_generator(register_a, register_b, ancilla):
    """
    Generate the swap test circuit primitive.

    Registers A and B must be of equivalent size for swap to work.  This module
    uses the CSWAP gate in pyquil.

    :param List register_a: qubit labels in the 'A' register
    :param List register_b: qubit labels in the 'B' register
    :param ancilla: ancilla to measure and control the swap operation.
    :return: Program
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


def run_swap_test(program_a, program_b, number_of_measurements, quantum_resource,
                  classical_memory=0, ancilla=None):
    """
    Run a swap test and return the calculated overlap

    :param Program program_a: state preparation for the 'A' register
    :param Program program_b: state preparation for the 'B' register
    :param Int number_of_measurements: number of times to measure
    :param quantum_resource: QAM connection object
    :param Int classical_memory: location of classical memory slot to use
    :param ancilla: Index of ancilla
    :return: overlap of states prepared by program_a and program_b
    :rtype: float
    """
    register_a = list(program_a.get_qubits())
    register_b = list(program_b.get_qubits())
    if ancilla is None:
        ancilla = max(register_a + register_b) + 1

    swap_test_program = Program()
    swap_test_program += program_a + program_b
    swap_test_program += swap_circuit_generator(register_a, register_b, ancilla)
    swap_test_program.measure(ancilla, [classical_memory])

    # TODO: instead of number of measurements have user set precision of overlap
    # estimate and calculate number of shots to be within this precision
    results = quantum_resource.run(swap_test_program,
                                   classical_memory,
                                   trials=number_of_measurements)

    probability_of_one = np.mean(results)
    if probability_of_one > 0.5:
        raise ValueError("measurements indicate overlap is negative")

    estimated_overlap = np.sqrt(1 - 2 * probability_of_one)
    return estimated_overlap





