"""Tests for utils"""

import pytest

from grove.alpha.arbitrary_state.arbitrary_state import *
from grove.pyqaoa.utils import compare_progs


@pytest.mark.skip(reason="Must add support for Forest connections in testing")
class TestCreateArbitraryState(object):
    def test_state_generation_simple_length_two(self):
        _state_generation_test_helper([1 + 1j, 2])

    def test_state_generation_simple_length_four(self):
        _state_generation_test_helper([1, 2, 3, 4])

    def test_state_generation_complex_length_five(self):
        _state_generation_test_helper([1 + 2j, 3 + 4j, -1 - 5j, 6 - 8j, 7])

    def test_state_generation_complex_length_eight(self):
        _state_generation_test_helper(
            [1 + 2j, 3 + 4j, -1 - 5j, 6 - 8j, 7, 2, 0.5j, 2])

    def test_state_generation_complex_length_huge(self):
        _state_generation_test_helper(np.array(range(50)) * 1j)

    def test_padded_zeros(self):
        _state_generation_test_helper([1, 0, 0])

    def test_single_one(self):
        _state_generation_test_helper([1], [4])

    def test_long_padded_zeros(self):
        _state_generation_test_helper([0.5j, 0.5, 0, 1, 0, 0, 0, 0, 0],
                                      range(3, 7))

    def test_forward_padded_zeros(self):
        _state_generation_test_helper([0, 0, 1])


class TestUniformlyControlledRotationMatrix(object):
    def test_one_control(self):
        expected = 1 / 2. * np.array([[1, 1],
                                      [1, -1]])
        actual = get_uniformly_controlled_rotation_matrix(1)
        assert np.allclose(expected, actual)

    def test_two_controls(self):
        expected = 1 / 4. * np.array([[1, 1, 1, 1],
                                      [1, -1, 1, -1],
                                      [1, -1, -1, 1],
                                      [1, 1, -1, -1]])
        actual = get_uniformly_controlled_rotation_matrix(2)

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
        actual = get_uniformly_controlled_rotation_matrix(3)

        assert np.allclose(expected, actual)


class TestUniformlyControlledCNOTPositions(object):
    def test_one_control(self):
        expected = [1, 1]
        actual = get_cnot_control_positions(1)
        assert expected == actual

    def test_two_controls(self):
        expected = [1, 2, 1, 2]
        actual = get_cnot_control_positions(2)
        assert expected == actual

    def test_three_controls(self):
        expected = [1, 2, 1, 3, 1, 2, 1, 3]
        actual = get_cnot_control_positions(3)
        assert expected == actual

    def test_four_controls(self):
        expected = [1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 4]
        actual = get_cnot_control_positions(4)
        assert expected == actual


class TestGetRotationParameters(object):
    def test_length_four_coefficients(self):
        phases = [np.pi / 2, np.pi / 6, -np.pi / 7, np.pi / 8]
        magnitudes = [0.1, 0.2, 0.3, 0.4]
        z_thetas, y_thetas, new_phases, new_magnitudes = \
            get_rotation_parameters(phases, magnitudes)
        expected_z_thetas = [np.pi / 3, -np.pi * 15 / 56]
        expected_y_thetas = [2 * np.arcsin(-0.1 / np.sqrt(0.1)),
                             2 * np.arcsin(-0.1 / np.sqrt(0.5))]
        expected_new_phases = [np.pi / 3, -np.pi / 112]
        expected_new_magnitudes = [np.sqrt(0.025), np.sqrt(0.125)]

        assert np.allclose(expected_z_thetas, z_thetas)
        assert np.allclose(expected_y_thetas, y_thetas)
        assert np.allclose(expected_new_phases, new_phases)
        assert np.allclose(expected_new_magnitudes, new_magnitudes)

    def test_length_eight_coefficients(self):
        phases = [-np.pi / 6, -np.pi / 6, np.pi / 13, -np.pi / 7,
                  -np.pi / 9, np.pi / 2, np.pi / 4, np.pi / 5]
        magnitudes = [0.7, 0.3, 0.25, 0.6,
                      0.84, 0.84, 0.05, 0.4]
        z_thetas, y_thetas, new_phases, new_magnitudes = \
            get_rotation_parameters(phases, magnitudes)
        expected_z_thetas = [0, np.pi * 20 / 91, -np.pi * 11 / 18, np.pi / 20]
        expected_y_thetas = [2 * np.arcsin(0.4 / np.sqrt(1.16)),
                             2 * np.arcsin(-0.35 / np.sqrt(0.845)),
                             0,
                             2 * np.arcsin(-0.35 / np.sqrt(0.325))]
        expected_new_phases = [-np.pi / 6, -np.pi * 3 / 91, np.pi * 7 / 36,
                               np.pi * 9 / 40]
        expected_new_magnitudes = [np.sqrt(0.29), np.sqrt(0.21125),
                                   0.84, np.sqrt(0.08125)]

        assert np.allclose(expected_z_thetas, z_thetas)
        assert np.allclose(expected_y_thetas, y_thetas)
        assert np.allclose(expected_new_phases, new_phases)
        assert np.allclose(expected_new_magnitudes, new_magnitudes)


class TestGetReversedUnificationProgram(object):
    def test_length_four_phase_rotations(self):
        angles = [np.pi / 4, np.pi / 11, np.pi / 7, -np.pi / 18]
        control_indices = get_cnot_control_positions(2)
        controls = [3, 8]
        target = 0
        reverse_prog = \
            get_reversed_unification_program(angles, control_indices,
                                             target, controls, "phase")

        expected_prog = pq.Program().inst(CNOT(8, 0)) \
            .inst(RZ(np.pi / 18, 0)) \
            .inst(CNOT(3, 0)) \
            .inst(RZ(-np.pi / 7, 0)) \
            .inst(CNOT(8, 0)) \
            .inst(RZ(-np.pi / 11, 0)) \
            .inst(CNOT(3, 0)) \
            .inst(RZ(-np.pi / 4, 0))

        compare_progs(reverse_prog, expected_prog)

    def test_length_eight_magnitude_rotations(self):
        angles = [0, - np.pi / 12, np.pi / 15, -np.pi / 6,
                  np.pi / 13, np.pi / 10, -np.pi / 3, np.pi / 4]
        control_indices = get_cnot_control_positions(3)
        controls = [1, 6, 2]
        target = 4
        reverse_prog = \
            get_reversed_unification_program(angles, control_indices,
                                             target, controls, "magnitude")

        expected_prog = pq.Program().inst(CNOT(2, 4)) \
            .inst(RY(-np.pi / 4, 4)) \
            .inst(CNOT(1, 4)) \
            .inst(RY(np.pi / 3, 4)) \
            .inst(CNOT(6, 4)) \
            .inst(RY(-np.pi / 10, 4)) \
            .inst(CNOT(1, 4)) \
            .inst(RY(-np.pi / 13, 4)) \
            .inst(CNOT(2, 4)) \
            .inst(RY(np.pi / 6, 4)) \
            .inst(CNOT(1, 4)) \
            .inst(RY(- np.pi / 15, 4)) \
            .inst(CNOT(6, 4)) \
            .inst(RY(np.pi / 12, 4)) \
            .inst(CNOT(1, 4))

        compare_progs(reverse_prog, expected_prog)


def _state_generation_test_helper(v, qubits=None):
    # encode vector in quantum state
    p = create_arbitrary_state(v, qubits)
    qvm = SyncConnection()
    wf, _ = qvm.wavefunction(p)

    # normalize and pad with zeros
    v_norm = v / np.linalg.norm(v)
    while len(v_norm) < len(wf.amplitudes):
        v_norm = np.append(v_norm, 0)

    # wavefunction amplitudes should resemble vector
    for pair in zip(v_norm, wf.amplitudes):
        assert np.allclose(*pair)
