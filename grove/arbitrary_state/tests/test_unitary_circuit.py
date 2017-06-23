"""Tests for utils"""

import numpy as np
import pytest
from grove.arbitrary_state.arbitrary_state import create_arbitrary_state
from pyquil.api import SyncConnection


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation_simple_length_two():
    _state_generation_test_helper([1+1j, 2])


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation_simple_length_four():
    _state_generation_test_helper([1, 2, 3, 4])


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation_complex_length_five():
    _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7])


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation_complex_length_eight():
    _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7, 2, 0.5j, 2])


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation_complex_length_huge():
    _state_generation_test_helper(np.array(range(50))*1j)


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_padded_zeros():
    _state_generation_test_helper([1, 0, 0])


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_single_one():
    _state_generation_test_helper([1], 4)


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_long_padded_zeros():
    _state_generation_test_helper([0.5j, 0.5, 0, 1, 0, 0, 0, 0, 0], 3)


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_forward_padded_zeros():
    _state_generation_test_helper([0, 0, 1])


def _state_generation_test_helper(v, offset=0):
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
