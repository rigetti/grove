from mock import patch
import pytest

import numpy as np

from grove.deutsch_jozsa.deutsch_jozsa import DeutschJosza


@pytest.mark.parametrize("bitmap, expected_bitstring",
                         [({"0": "0", "1": "0"},
                          np.asarray([0, 0], dtype=int)),
                          ({"0": "1", "1": "1"},
                          np.asarray([0, 0], dtype=int))])
def test_deutsch_jozsa_one_qubit_exact_zeros(bitmap, expected_bitstring):
    dj = DeutschJosza()
    with patch("pyquil.api.SyncConnection") as qvm:
        qvm.run_and_measure.return_value = expected_bitstring
        is_constant = dj.is_constant(qvm, bitmap)
    assert is_constant


def test_deutsch_jozsa_one_qubit_balanced():
    balanced_one_qubit_bitmap = {"0": "0", "1": "1"}
    dj = DeutschJosza()
    with patch("pyquil.api.SyncConnection") as qvm:
        # Should just be not the zero vector
        expected_bitstring = np.asarray([0, 1], dtype=int)
        qvm.run_and_measure.return_value = expected_bitstring
        is_constant = dj.is_constant(qvm, balanced_one_qubit_bitmap)
    assert not is_constant


def test_deutsch_jozsa_two_qubit_neither():
    exact_one_qubit_bitmap = {"00": "0", "01": "0", "10": "1", "11": "00"}
    dj = DeutschJosza()
    with pytest.raises(ValueError):
        with patch("pyquil.api.SyncConnection") as qvm:
            _ = dj.is_constant(qvm, exact_one_qubit_bitmap)
