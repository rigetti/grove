"""
Tests on the swap test in the circuit_primitives module
"""
import numpy as np
import pytest
from unittest.mock import patch
from pyquil import Program
from pyquil.gates import CSWAP, H

from grove.circuit_primitives.swap import (swap_circuit_generator,
                                           run_swap_test,
                                           RegisterSizeMismatch)

def test_swap_circuit_gen_type():
    """
    Test the type checking
    """
    with pytest.raises(TypeError):
        swap_circuit_generator(5, [1, 2], 0)

    with pytest.raises(TypeError):
        swap_circuit_generator([1, 2], 5, 0)

    with pytest.raises(RegisterSizeMismatch):
        swap_circuit_generator([1, 2], [3], 0)


def test_default_ancilla_assignment():
    """
    Make sure ancilla is assigned to max(regA + regB) + 1 by default
    :return:
    """
    test_prog_for_ancilla = swap_circuit_generator([1, 2], [5, 6], None)
    instruction = test_prog_for_ancilla.pop()
    assert instruction.qubits[0].index == 7


def test_cswap_program():
    """
    Test if the correct program is returned.  Half way to system test
    """
    test_prog = swap_circuit_generator([1, 2], [5, 6], None)
    true_prog = Program()
    true_prog += H(7)
    true_prog += CSWAP(7, 1, 5)
    true_prog += CSWAP(7, 2, 6)
    true_prog += H(7)

    assert test_prog.out() == true_prog.out()


def test_run_swap():
    """
    Test the qvm return piece
    """
    expected_bitstring = [1, 1, 1, 0, 0, 0, 0, 0, 0]
    prog_a = Program().inst(H(0))
    prog_b = Program().inst(H(1))
    with patch("pyquil.api.QuantumComputer") as qc:
        qc.run.return_value = expected_bitstring
        test_overlap = run_swap_test(prog_a, prog_b,
                                     number_of_measurements=5,
                                     quantum_resource=qc)

        assert np.isclose(np.sqrt(1 - 2 * np.mean(expected_bitstring)),
                          test_overlap)

    expected_bitstring = [1, 1, 1, 0, 1]
    prog_a = Program().inst(H(0))
    prog_b = Program().inst(H(1))
    with patch("pyquil.api.QuantumComputer") as qc:
        qc.run.return_value = expected_bitstring
        with pytest.raises(ValueError):
            test_overlap = run_swap_test(prog_a, prog_b,
                                         number_of_measurements=5,
                                         quantum_resource=qc)

