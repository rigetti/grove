"""Tests for utils"""

import pytest

from grove.arbitrary_state.arbitrary_state import *


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestCreateArbitraryState(object):
    def test_state_generation_simple_length_two(self):
        _state_generation_test_helper([1+1j, 2])
    
    def test_state_generation_simple_length_four(self):
        _state_generation_test_helper([1, 2, 3, 4])

    def test_state_generation_complex_length_five(self):
        _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7])
    
    def test_state_generation_complex_length_eight(self):
        _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7, 2, 0.5j, 2])
    
    def test_state_generation_complex_length_huge(self):
        _state_generation_test_helper(np.array(range(50))*1j)
    
    def test_padded_zeros(self):
        _state_generation_test_helper([1, 0, 0])
    
    def test_single_one(self):
        _state_generation_test_helper([1], 4)
    
    def test_long_padded_zeros(self):
        _state_generation_test_helper([0.5j, 0.5, 0, 1, 0, 0, 0, 0, 0], 3)
    
    def test_forward_padded_zeros(self):
        _state_generation_test_helper([0, 0, 1])


class TestUniformlyControlledRotationMatrix(object):
    def test_one_control(self):
        expected = 1/2.*np.array([[1, 1],
                                  [1, -1]])
        actual = uniformly_controlled_rotation_matrix(1)
        assert np.allclose(expected, actual)

    def test_two_controls(self):
        expected = 1/4.*np.array([[1, 1, 1, 1],
                                  [1, -1, 1, -1],
                                  [1, -1, -1, 1],
                                  [1, 1, -1, -1]])
        actual = uniformly_controlled_rotation_matrix(2)

        assert np.allclose(expected, actual)

    def test_three_controls(self):
        expected = 1 / 8. * np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                      [1, -1, 1, -1, 1, -1, 1, -1],
                                      [1, -1, -1, 1, 1, -1, -1, 1],
                                      [1, 1, -1, -1, 1, 1, -1, -1],
                                      [1, 1, -1, -1, -1, -1, 1, 1],
                                      [1, -1, -1, 1, -1, 1, 1, -1],
                                      [1, -1, 1, -1, -1, 1, -1, 1],
                                      [1, 1, 1, 1, -1, -1, -1, -1]])
        actual = uniformly_controlled_rotation_matrix(3)

        assert np.allclose(expected, actual)


class TestUniformlyControlledCNOTPositions(object):
    def test_one_control(self):
        expected = [1, 1]
        actual = uniformly_controlled_cnot_control_positions(1)
        assert expected == actual

    def test_two_controls(self):
        expected = [1, 2, 1, 2]
        actual = uniformly_controlled_cnot_control_positions(2)
        assert expected == actual

    def test_three_controls(self):
        expected = [1, 2, 1, 3, 1, 2, 1, 3]
        actual = uniformly_controlled_cnot_control_positions(3)
        assert expected == actual

    def test_four_controls(self):
        expected = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 4]
        actual = uniformly_controlled_cnot_control_positions(4)
        assert expected == actual


def _state_generation_test_helper(v, offset=0):
    # encode vector in quantum state
    p = create_arbitrary_state(v, offset)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)

    # normalize and pad with zeros
    v_norm = v / np.linalg.norm(v)
    while len(v_norm) < len(wf.amplitudes):
        v_norm = np.append(v_norm, 0)

    # wavefunction amplitudes should resemble vector
    for pair in zip(v_norm, wf.amplitudes):
        assert np.allclose(*pair)
