import pytest
from grove.state_generation.state_generation import generate_state, unitary_operator, next_power_of_two
import pyquil.quil as pq
from pyquil.api import SyncConnection
import numpy as np

def test_unitary():
    length = 30
    # make an arbitrary complex vector
    v = np.random.uniform(-1, 1, length) + np.random.uniform(-1, 1, length) * 1j

    # normalize and pad with zeros
    v_norm = v / np.linalg.norm(v)
    while len(v_norm) < next_power_of_two(len(v)):
        v_norm = np.append(v_norm, 0)

    # generate unitary operator
    U = unitary_operator(v_norm)

    # make sure U|0> = |v>
    zero_state = np.zeros(len(U))
    zero_state[0] = 1
    assert np.allclose(U.dot(zero_state), v_norm)

@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation():
    length = 30
    # make an arbitrary complex vector
    v = np.random.uniform(-1, 1, length) + np.random.uniform(-1, 1, length)*1j

    # encode vector in quantum state
    p = generate_state(v)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)

    # normalize and pad with zeros
    v_norm = v / np.linalg.norm(v)
    while len(v_norm) < len(wf.amplitudes):
        v_norm = np.append(v_norm, 0)

    # wavefunction amplitudes should resemble vector
    assert np.allclose(v_norm, wf.amplitudes)
