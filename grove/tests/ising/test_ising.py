from grove.ising.ising_qaoa import ising_qaoa
from grove.ising.ising_qaoa import energy_value
import numpy as np
from mock import patch


def test_energy_value():
    J = {(0, 1): 2.3}
    h = {0: -2.4, 1: 5.2}
    sol = [1, -1]
    ener_ising = energy_value(h, J, sol)

    assert(np.isclose(ener_ising, -9.9))


def test_ising_mock():
    with patch("pyquil.api.QVMConnection") as cxn:
        # Mock the response
        cxn.run_and_measure.return_value = [[1, 1, 0, 1]]
        cxn.expectation.return_value = [-0.4893891813015294, 0.8876822987380573, -0.4893891813015292, -0.9333372094534063, -0.9859245403423198, 0.9333372094534065]

    J = {(0, 1): -2, (2, 3): 3}
    h = {0: 1, 1: 1, 2: -1, 3: 1}
    p = 1
    most_freq_string_ising, energy_ising, circuit = ising_qaoa(h, J, num_steps=p, vqe_option=None, connection=cxn)

    assert most_freq_string_ising == [-1, -1, 1, -1]
    assert energy_ising == -9
