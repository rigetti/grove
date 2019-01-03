import numpy as np
import pytest
from unittest.mock import patch
from pyquil import Program
from pyquil.gates import H, X
from pyquil.quilatom import QubitPlaceholder

from grove.amplification.amplification import HADAMARD_DIFFUSION_LABEL
from grove.amplification.grover import Grover

identity_oracle = Program()
"""Does nothing on all inputs."""


def check_instructions(intended_instructions, actual_instructions):
    """Checks if two sequences of instructions are the same.

    This is useful because Program equality cares about labels.
    """
    assert len(intended_instructions) == len(actual_instructions)
    for i, instruction in enumerate(actual_instructions):
        qubits = instruction.qubits
        intended_gate = intended_instructions[i]
        if isinstance(intended_instructions[i], str):
            assert instruction.name == intended_gate
        else:
            assert instruction.name == intended_gate(*qubits).name


@pytest.fixture()
def x_oracle():
    """Applies an X gate to the ancilla on all inputs. Returns the program, and the query Qubit."""
    p = Program()
    qubit = QubitPlaceholder()
    p.inst(X(qubit))
    return p, qubit


def test_trivial_grover():
    """Testing that we construct the correct circuit for Grover's Algorithm with one step, and the
     identity_oracle on one qubit.
     """
    qubits = [0]
    gates = [H, H, HADAMARD_DIFFUSION_LABEL, H]
    generated_trivial_grover = Grover().oracle_grover(identity_oracle, qubits, 1)
    check_instructions(gates, generated_trivial_grover)


def test_x_oracle_one_grover(x_oracle):
    """Testing that Grover's algorithm with an oracle that applies an X gate to the query bit works,
     with one iteration."""
    qubits = [0]
    oracle, _ = x_oracle
    generated_x_oracle_grover = Grover().oracle_grover(oracle, qubits, num_iter=1)
    gates = [H, X, H, HADAMARD_DIFFUSION_LABEL, H]
    check_instructions(gates, generated_x_oracle_grover)


def test_x_oracle_two_grover(x_oracle):
    """Testing that Grover's algorithm with an oracle that applies an X gate to the query bit works,
     with two iterations."""
    qubits = [0]
    oracle, _ = x_oracle
    generated_x_oracle_grover = Grover().oracle_grover(oracle, qubits, num_iter=2)
    # First we put the input into uniform superposition.
    gates = [H]
    for _ in range(2):
        # Now an oracle is applied.
        gates.append(X)
        # We apply the diffusion operator.
        gates.append(H)
        gates.append(HADAMARD_DIFFUSION_LABEL)
        gates.append(H)
    check_instructions(gates, generated_x_oracle_grover)


def test_optimal_grover(x_oracle):
    """Testing that Grover's algorithm with an oracle that applies an X gate to the query bit works,
     and defaults to the optimal number of iterations."""
    qubits = [0]
    oracle, _ = x_oracle
    generated_one_iter_grover = Grover().oracle_grover(oracle, qubits)
    # First we put the input into uniform superposition.
    gates = [H, X, H, HADAMARD_DIFFUSION_LABEL, H]
    check_instructions(gates, generated_one_iter_grover)


def test_find_bitstring():
    bitstring_map = {"0": 1, "1": -1}
    builder = Grover()
    with patch("pyquil.api.QuantumComputer") as qc:
        expected_bitstring = [0, 1]
        qc.run.return_value = ["".join([str(bit) for bit in expected_bitstring])]
    returned_bitstring = builder.find_bitstring(qc, bitstring_map)
    prog = builder.grover_circuit
    # Make sure it only defines the ORACLE gate and the DIFFUSION gate.
    assert len(prog.defined_gates) == 2
    # Make sure that it produces the oracle we expect.
    assert (prog.defined_gates[0].matrix == np.array([[1, 0], [0, -1]])).all()
    expected_bitstring = "".join([str(bit) for bit in expected_bitstring])
    returned_bitstring = "".join([str(bit) for bit in returned_bitstring])
    assert expected_bitstring == returned_bitstring
