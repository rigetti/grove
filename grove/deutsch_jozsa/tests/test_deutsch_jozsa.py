from grove.deutsch_jozsa.deutsch_jozsa import *
from pyquil.api import SyncConnection
import pytest

def run(n, mappings):
    deutsch_program = pq.Program()
    qubits = [deutsch_program.alloc() for _ in range(n)]
    ancilla = deutsch_program.alloc()
    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancilla)
    deutsch_program += deutsch_jozsa(oracle, qubits, ancilla)
    deutsch_program.out()
    qvm = SyncConnection()
    results = qvm.run_and_measure(deutsch_program, [q.index() for q in qubits])
    return "balanced" if 1 in results[0] else "constant"

@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_balanced():
    n = 3
    mappings = [0, 1, 1, 0, 1, 0, 0, 1]
    assert run(n, mappings) == "balanced"

@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_constant():
    n = 3
    mappings_0 = [0] * 8
    mappings_1 = [1] * 8
    assert run(n, mappings_0) == "constant"
    assert run(n, mappings_1) == "constant"
