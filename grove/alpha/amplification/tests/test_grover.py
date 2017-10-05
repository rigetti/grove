import pytest
from mock import Mock, patch

from pyquil.quil import Program
from pyquil.gates import X, Z, H
from pyquil.quilbase import Qubit

from grove.alpha.amplification.grover import grover
from grove.pyquil_utilities import prog_equality, synthesize_programs

BASE_PATH=""

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
    generated_trivial_grover = grover(identity_oracle, qubits, 1)
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
    generated_x_oracle_grover = grover(oracle, qubits, 1)
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
    generated_x_oracle_grover = grover(oracle, qubits, 2)
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


def test_grover_bad_input():
    with pytest.raises(ValueError):
        _ = grover(identity_oracle, [], 2)
    with pytest.raises(ValueError):
        _ = grover(identity_oracle, ["foo"], 2)
