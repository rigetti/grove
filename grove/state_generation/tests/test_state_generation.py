import pytest
from grove.state_generation.state_generation import generate_state
import pyquil.quil as pq
from pyquil.api import SyncConnection
import numpy as np

@pytest.mark.skip(reason="Must add support for Forest connections in testing")
def test_state_generation():
    # make an arbitrary complex vector
    v = [1+2j, 3+4j, -1-5j, 6-8j, 7]

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