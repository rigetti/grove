"""Tests for utils"""

from grove.unitary_circuit.utils import *
from pyquil.api import SyncConnection


def test_state_generation_simple_length_two():
    _state_generation_test_helper([1+1j, 2])


def test_state_generation_simple_length_four():
    _state_generation_test_helper([1, 2, 3, 4])


def test_state_generation_complex_length_five():
    _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7])


def test_state_generation_complex_length_eight():
    _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7, 2, 0.5j, 2])


def test_state_generation_complex_length_huge():
    _state_generation_test_helper(np.array(range(50))*1j)


def _state_generation_test_helper(v):
    # encode vector in quantum state
    p = create_arbitrary_state(v)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)

    # normalize and pad with zeros
    v_norm = v / np.linalg.norm(v)
    while len(v_norm) < len(wf.amplitudes):
        v_norm = np.append(v_norm, 0)

    # wavefunction amplitudes should resemble vector
    for pair in zip(v_norm, wf.amplitudes):
        assert np.allclose(*pair)

if __name__ == "__main__":
    v = list(input("Give your array: "))
    p = create_arbitrary_state(v)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)
    print list(v / np.linalg.norm(v))
    print wf
    print p.out()