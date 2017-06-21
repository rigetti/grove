"""Tests for utils"""

from grove.unitary_circuit.utils import *
from pyquil.api import SyncConnection


def test_state_generation_simple_length_two():
    state_generation_test_helper([1+1j, 2])


def test_state_generation_simple_length_four():
    state_generation_test_helper([1, 2, 3, 4])


def test_state_generation_complex_length_five():
    state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7])


def state_generation_test_helper(v):
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


def debug_create_arbitrary_state(lst):
    p = create_arbitrary_state(lst, True)
    p.out()
    cxn = SyncConnection()
    wf, _ = cxn.wavefunction(p)
    print wf


def debug_n_bit_control():
    p = pq.Program()
    qvm = SyncConnection()

    p.inst(map(X, range(2)))
    wf, _ = qvm.wavefunction(p)
    print wf

    p = n_qubit_controlled_RY(range(2), 1, np.pi)
    wf, _ = qvm.wavefunction(p)
    print wf
    print p.out()

if __name__=='__main__':
    lst = [1+2j, 3+4j, -1-5j, 6-8j, 7]
    #lst = [1, 2, 3, 4]
    debug_create_arbitrary_state(lst)
    #test_n_bit_control()