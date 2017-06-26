"""Tests for utils"""

import pytest

from grove.arbitrary_state.arbitrary_state import *


class TestCreateArbitraryState(object):
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_state_generation_simple_length_two(self):
        _state_generation_test_helper([1+1j, 2])
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_state_generation_simple_length_four(self):
        _state_generation_test_helper([1, 2, 3, 4])
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_state_generation_complex_length_five(self):
        _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7])
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_state_generation_complex_length_eight(self):
        _state_generation_test_helper([1+2j, 3+4j, -1-5j, 6-8j, 7, 2, 0.5j, 2])
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_state_generation_complex_length_huge(self):
        _state_generation_test_helper(np.array(range(50))*1j)
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_padded_zeros(self):
        _state_generation_test_helper([1, 0, 0])
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_single_one(self):
        _state_generation_test_helper([1], 4)
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
    def test_long_padded_zeros(self):
        _state_generation_test_helper([0.5j, 0.5, 0, 1, 0, 0, 0, 0, 0], 3)
    
    
    @pytest.mark.skip(reason="Must add support for Forest connections in testing")
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
        print expected
        print actual

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
        print expected
        print actual

        assert np.allclose(expected, actual)


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
