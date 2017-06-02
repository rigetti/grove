from grove.deutsch_jozsa.deutsch_jozsa import *
import pyquil.forest as forest

def run(n, mappings):
    deutsch_program = pq.Program()
    qubits = [deutsch_program.alloc() for _ in range(n)]
    ancilla = deutsch_program.alloc()
    scratch_bit = deutsch_program.alloc()
    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancilla, scratch_bit)
    deutsch_program += deutsch_jozsa(oracle, qubits, ancilla)
    deutsch_program.out()
    qvm = forest.Connection()
    results = qvm.run_and_measure(deutsch_program, [q.index() for q in qubits])
    return "balanced" if 1 in results[0] else "constant"

def test_balanced():
    n = 3
    mappings = [0, 1, 1, 0, 1, 0, 0, 1]
    assert run(n, mappings) == "balanced"

def test_constant():
    n = 3
    mappings_0 = [0] * 8
    mappings_1 = [1] * 8
    assert run(n, mappings_0) == "constant"
    assert run(n, mappings_1) == "constant"