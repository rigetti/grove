import numpy as np
import pytest
from mock import patch
from pyquil.gates import X, Z, H
from pyquil.quil import Program

from grove.amplification.grover import Grover
from grove.tests.utils.utils_for_testing import prog_equality, synthesize_programs

identity_oracle = Program()
"""Does nothing on all inputs."""


@pytest.fixture()
def x_oracle():
    """Applies an X gate to the ancilla on all inputs. Returns the program, and the query Qubit."""
    p = Program()
    qubit = p.alloc()
    p.inst(X(qubit))
    return p, qubit


def test_trivial_grover():
    """Testing that we construct the correct circuit for Grover's Algorithm with one step, and the
     identity_oracle on one qubit.
     """
    trivial_grover = Program()
    qubit0 = trivial_grover.alloc()
    # First we put the input into uniform superposition.
    trivial_grover.inst(H(qubit0))
    # No oracle is applied, so we just apply the diffusion operator.
    trivial_grover.inst(H(qubit0))
    trivial_grover.inst(Z(qubit0))
    trivial_grover.inst(H(qubit0))
    qubits = [qubit0]
    generated_trivial_grover = Grover().oracle_grover(identity_oracle, qubits, 1)
    generated_trivial_grover.synthesize()
    trivial_grover.synthesize()
    assert prog_equality(generated_trivial_grover, trivial_grover)


def test_x_oracle_one_grover(x_oracle):
    """Testing that Grover's algorithm with an oracle that applies an X gate to the query bit works,
     with one iteration."""
    x_oracle_grover = Program()
    qubit0 = x_oracle_grover.alloc()
    qubits = [qubit0]
    oracle, query_qubit = x_oracle
    with patch("pyquil.quilbase.InstructionGroup.alloc") as mock_alloc:
        mock_alloc.return_value = qubit0
    generated_x_oracle_grover = Grover().oracle_grover(oracle, qubits, 1)
    # First we put the input into uniform superposition.
    x_oracle_grover.inst(H(qubit0))
    # Now an oracle is applied.
    x_oracle_grover.inst(X(query_qubit))
    # We now apply the diffusion operator.
    x_oracle_grover.inst(H(qubit0))
    x_oracle_grover.inst(Z(qubit0))
    x_oracle_grover.inst(H(qubit0))
    synthesize_programs(generated_x_oracle_grover, x_oracle_grover)
    assert prog_equality(generated_x_oracle_grover, x_oracle_grover)


def test_x_oracle_two_grover(x_oracle):
    """Testing that Grover's algorithm with an oracle that applies an X gate to the query bit works,
     with two iterations."""
    x_oracle_grover = Program()
    qubit0 = x_oracle_grover.alloc()
    qubits = [qubit0]
    oracle, query_qubit = x_oracle
    with patch("pyquil.quilbase.InstructionGroup.alloc") as mock_alloc:
        mock_alloc.return_value = qubit0
    generated_x_oracle_grover = Grover().oracle_grover(oracle, qubits, 2)
    # First we put the input into uniform superposition.
    x_oracle_grover.inst(H(qubit0))
    # Two iterations.
    for _ in range(2):
        # Now an oracle is applied.
        x_oracle_grover.inst(X(query_qubit))
        # We apply the diffusion operator.
        x_oracle_grover.inst(H(qubit0))
        x_oracle_grover.inst(Z(qubit0))
        x_oracle_grover.inst(H(qubit0))
    synthesize_programs(generated_x_oracle_grover, x_oracle_grover)
    assert prog_equality(generated_x_oracle_grover, x_oracle_grover)


def test_optimal_grover(x_oracle):
    """Testing that Grover's algorithm with an oracle that applies an X gate to the query bit works,
     and defaults to the optimal number of iterations."""
    grover_precircuit = Program()
    qubit0 = grover_precircuit.alloc()
    qubits = [qubit0]
    oracle, query_qubit = x_oracle
    with patch("pyquil.quilbase.InstructionGroup.alloc") as mock_alloc:
        mock_alloc.return_value = qubit0
        generated_one_iter_grover = Grover().oracle_grover(oracle, qubits)
    # First we put the input into uniform superposition.
    grover_precircuit.inst(H(qubit0))
    # We only do one iteration, which is the result of rounding pi * sqrt(N)/4
    iter = Program()

    # An oracle is applied.
    iter.inst(X(query_qubit))
    # We now apply the diffusion operator.
    iter.inst(H(qubit0))
    iter.inst(Z(qubit0))
    iter.inst(H(qubit0))
    one_iter_grover = grover_precircuit + iter
    synthesize_programs(generated_one_iter_grover, one_iter_grover)
    assert prog_equality(generated_one_iter_grover, one_iter_grover)


def test_find_bistring():
    bitstring_map = {"0": 1, "1": -1}
    builder = Grover()
    with patch("pyquil.api.SyncConnection") as qvm:
        expected_bitstring = [0, 1]
        qvm.run_and_measure.return_value = [expected_bitstring, ]
    returned_bitstring = builder.find_bitstring(qvm, bitstring_map)
    prog = builder.grover_circuit
    # Make sure it only defines the one ORACLE gate.
    assert len(prog.defined_gates) == 1
    # Make sure that it produces the oracle we expect.
    assert (prog.defined_gates[0].matrix == np.array([[1, 0], [0, -1]])).all()
    expected_bitstring = "".join([str(bit) for bit in expected_bitstring])
    returned_bitstring = "".join([str(bit) for bit in returned_bitstring])
    assert expected_bitstring == returned_bitstring
