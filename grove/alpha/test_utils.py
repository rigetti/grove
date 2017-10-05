from pyquil.quil import Program

from grove.alpha.utils import is_valid_qubits


def test_is_valid_qubits():
    prog = Program()
    qubit = prog.alloc()
    assert is_valid_qubits([qubit])
    assert not is_valid_qubits(qubit)
    assert is_valid_qubits([1])
    assert is_valid_qubits([1, qubit])
