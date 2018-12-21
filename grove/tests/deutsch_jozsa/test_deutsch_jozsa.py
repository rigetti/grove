import numpy as np
import pytest
from unittest.mock import patch
from pyquil import Program
from pyquil.gates import X, H, CNOT

from grove.deutsch_jozsa.deutsch_jozsa import DeutschJosza, ORACLE_GATE_NAME


@pytest.mark.parametrize("bitmap, expected_bitstring",
                         [({"0": "0", "1": "0"},
                          np.asarray([0, 0], dtype=int)),
                          ({"0": "1", "1": "1"},
                          np.asarray([0, 0], dtype=int))])
def test_deutsch_jozsa_one_qubit_exact_zeros(bitmap, expected_bitstring):
    dj = DeutschJosza()
    with patch("pyquil.api.QuantumComputer") as qc:
        qc.run.return_value = expected_bitstring
        is_constant = dj.is_constant(qc, bitmap)
    assert is_constant


def test_deutsch_jozsa_one_qubit_balanced():
    balanced_one_qubit_bitmap = {"0": "0", "1": "1"}
    dj = DeutschJosza()
    with patch("pyquil.api.QuantumComputer") as qc:
        # Should just be not the zero vector
        expected_bitstring = np.asarray([0, 1], dtype=int)
        qc.run.return_value = expected_bitstring
        is_constant = dj.is_constant(qc, balanced_one_qubit_bitmap)
    assert not is_constant


def test_deutsch_jozsa_two_qubit_neither():
    exact_two_qubit_bitmap = {"00": "0", "01": "0", "10": "1", "11": "00"}
    dj = DeutschJosza()
    with pytest.raises(ValueError):
        with patch("pyquil.api.QuantumComputer") as qc:
            _ = dj.is_constant(qc, exact_two_qubit_bitmap)


def test_one_qubit_exact_zeros_circuit():
    exact_one_qubit_bitmap = {"0": "0", "1": "1"}
    dj = DeutschJosza()
    with patch("pyquil.api.QuantumComputer") as qc:
        # Should just be not the zero vector
        expected_bitstring = np.asarray([0, 1], dtype=int)
        qc.run.return_value = expected_bitstring
        _ = dj.is_constant(qc, exact_one_qubit_bitmap)
    # Ordering doesn't matter, so we pop instructions from a set
    expected_prog = Program()

    # We've defined the oracle and its dagger.
    dj_circuit = dj.deutsch_jozsa_circuit
    assert len(dj_circuit.defined_gates) == 2
    defined_oracle = None
    defined_oracle_inv = None
    for gate in dj_circuit.defined_gates:
        if gate.name == ORACLE_GATE_NAME:
            defined_oracle = gate
        else:
            defined_oracle_inv = gate

    assert defined_oracle is not None
    assert defined_oracle_inv is not None

    expected_prog.defgate(defined_oracle.name, defined_oracle.matrix)
    expected_prog.defgate(defined_oracle_inv.name, defined_oracle_inv.matrix)
    expected_prog.inst([X(1),
                        H(1),
                        H(0),
                        (defined_oracle.name, 2, 0),
                        CNOT(0, 1),
                        (defined_oracle_inv.name, 2, 0),
                        H(0)])
    assert expected_prog.out() == dj.deutsch_jozsa_circuit.out()
