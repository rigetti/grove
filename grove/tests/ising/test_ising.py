import numpy as np
from unittest.mock import patch, Mock
from pyquil.api import WavefunctionSimulator

from grove.ising.ising_qaoa import energy_value, ising


def test_energy_value():
    J = {(0, 1): 2.3}
    h = [-2.4, 5.2]
    sol = [1, -1]
    ener_ising = energy_value(h, J, sol)

    assert(np.isclose(ener_ising, -9.9))


def test_ising_mock():
    J = {(0, 1): -2, (2, 3): 3}
    h = [1, 1, -1, 1]
    p = 1

    with patch("pyquil.api.QuantumComputer") as fake_qc, \
            patch("grove.pyvqe.vqe.WavefunctionSimulator") as fake_wf:
        # Mock the response
        fake_qc.run.return_value = [[1, 1, 0, 1]]

        fake_wfs = Mock(WavefunctionSimulator)
        fake_wfs.expectation.return_value = [-0.4893891813015294,
                                             0.8876822987380573,
                                             -0.4893891813015292,
                                             -0.9333372094534063,
                                             -0.9859245403423198,
                                             0.9333372094534065]
        fake_wf.return_value = fake_wfs

        most_freq_string_ising, energy_ising, circuit = ising(h, J,
                                                              num_steps=p,
                                                              vqe_option=None,
                                                              connection=fake_qc)

    assert most_freq_string_ising == [-1, -1, 1, -1]
    assert energy_ising == -9
