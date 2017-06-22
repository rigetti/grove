"""Tests for utils"""

from grove.unitary_circuit.arbitrary_state import create_arbitrary_state
import numpy as np
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

def test_padded_zeros():
    _state_generation_test_helper([1, 0, 0])

def test_single_one():
    _state_generation_test_helper([1])

def test_long_padded_zeros():
    _state_generation_test_helper([0.5j, 0.5, 0, 1, 0, 0, 0, 0, 0])


def test_forward_padded_zeros():
    _state_generation_test_helper([0, 0, 1])


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
