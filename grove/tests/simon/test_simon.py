"""Test class for helper methods found simon"""

from os.path import abspath, dirname

import numpy as np
from mock import patch
from pyquil.quil import Program

from grove.simon.simon import Simon, create_1to1_bitmap, create_valid_2to1_bitmap

package_path = abspath(dirname(dirname(__file__)))

EXPECTED_SIMON_ORACLE = np.load(package_path + '/simon/data/simon_test_oracle.npy')


def _create_expected_program():
    expected_prog = Program()
    expected_prog.defgate("SIMON_ORACLE", EXPECTED_SIMON_ORACLE)
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    expected_prog.inst("H 2")

    expected_prog.inst("SIMON_ORACLE 5 4 3 2 1 0")

    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    expected_prog.inst("H 2")
    return expected_prog


def test_simon_class():
    """Test is based on worked example of Watrous lecture
    https://cs.uwaterloo.ca/~watrous/CPSC519/LectureNotes/06.pdf"""
    simon_algo = Simon()

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.side_effect = [
            (np.asarray([1, 1, 1], dtype=int), ),
            (np.asarray([1, 1, 1], dtype=int), ),
            (np.asarray([1, 0, 0], dtype=int), ),
            (np.asarray([1, 1, 1], dtype=int), ),
            (np.asarray([0, 0, 0], dtype=int), ),
            (np.asarray([0, 1, 1], dtype=int), ),
        ]

    bit_string_mapping = {
        '000': '101',
        '001': '010',
        '010': '000',
        '011': '110',

        '100': '000',
        '101': '110',
        '110': '101',
        '111': '010'
    }

    mask = simon_algo.find_mask(qvm, bit_string_mapping)

    assert simon_algo.n_qubits == 3
    assert simon_algo.n_ancillas == 3
    assert simon_algo._qubits == [0, 1, 2, 3, 4, 5]
    assert simon_algo.computational_qubits == [0, 1, 2]
    assert simon_algo.ancillas == [3, 4, 5]

    assert mask == [1, 1, 0]
    assert simon_algo.simon_circuit.__str__() == _create_expected_program().__str__()


def test_unitary_function_return():
    simon_algo = Simon()
    bit_string_mapping = {
        '000': '101',
        '001': '010',
        '010': '000',
        '011': '110',

        '100': '000',
        '101': '110',
        '110': '101',
        '111': '010'
    }

    actual_return = simon_algo._compute_unitary_oracle_matrix(bit_string_mapping)
    np.testing.assert_equal(actual_return[0], EXPECTED_SIMON_ORACLE)


def test_unitary_oracle_func_computer():
    bit_string_mapping = {
            '0': '1',
            '1': '0',
        }
    np.testing.assert_equal(Simon()._compute_unitary_oracle_matrix(bit_string_mapping)[0],
                            [[0., 0., 1., 0.],
                             [0., 1., 0., 0.],
                             [1., 0., 0., 0.],
                             [0., 0., 0., 1.]]
                            )


def test_unitary_oracle_func_computer_2():
    bit_string_mapping = {
            '00': '10',
            '01': '11',
            '10': '00',
            '11': '01'
        }
    np.testing.assert_equal(Simon()._compute_unitary_oracle_matrix(bit_string_mapping)[0],
                            [[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]
                            )


def test_no_substitution():
    simon_algo = Simon()
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        0: [1, 0, 1, 0, 0],
        1: [0, 1, 0, 0, 0],
        3: [0, 0, 0, 1, 0]
    }
    z = np.array([1, 1, 1, 0, 0])  # linear combination of first two rows hence won't add

    simon_algo._add_to_dict_of_indep_bit_vectors(z)
    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors

    W_expected = {
        0: [1, 0, 1, 0, 0],
        1: [0, 1, 0, 0, 0],
        3: [0, 0, 0, 1, 0]
    }

    np.testing.assert_equal(W_actual, W_expected)


def test_simple_conflict():
    simon_algo = Simon()
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        0: [1, 0, 1, 0, 0],
        1: [0, 1, 0, 0, 0],
        3: [0, 0, 0, 1, 0]
    }
    z = np.array([1, 0, 0, 0, 1])  # conflict with first row.

    simon_algo._add_to_dict_of_indep_bit_vectors(z)
    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors

    W_expected = {
        0: [1, 0, 1, 0, 0],
        1: [0, 1, 0, 0, 0],
        2: [0, 0, 1, 0, 1],
        3: [0, 0, 0, 1, 0]
    }

    np.testing.assert_equal(W_actual, W_expected)


def test_insert_directly():
    simon_algo = Simon()
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        0: [1, 1, 0, 0, 0],
        1: [0, 1, 0, 1, 0]
    }
    z = np.array([0, 0, 1, 0, 1])

    simon_algo._add_to_dict_of_indep_bit_vectors(z)
    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors
    W_expected = {
        0: [1, 1, 0, 0, 0],
        1: [0, 1, 0, 1, 0],
        2: [0, 0, 1, 0, 1]
    }

    np.testing.assert_equal(W_actual, W_expected)


def test_insert_after_xor():
    simon_algo = Simon()
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        0: [1, 0, 0, 0, 0, 0],
        1: [0, 1, 1, 0, 0, 0]
    }

    z = np.array([0, 0, 1, 0, 1, 1])

    simon_algo._add_to_dict_of_indep_bit_vectors(z)
    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors
    W_expected = {
        0: [1, 0, 0, 0, 0, 0],
        1: [0, 1, 1, 0, 0, 0],
        2: [0, 0, 1, 0, 1, 1]
    }

    np.testing.assert_equal(W_actual, W_expected)


def test_add_row_at_top():
    simon_algo = Simon()
    simon_algo.n_qubits = 4
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        1: [0, 1, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 1]
    }
    insert_row_num = simon_algo._add_missing_msb_vector()

    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors
    W_expected = {
        0: [1, 0, 0, 0],
        1: [0, 1, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 1]
    }

    assert insert_row_num == 0

    np.testing.assert_equal(W_actual, W_expected)


def test_add_row_at_bottom():
    simon_algo = Simon()
    simon_algo.n_qubits = 4
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        0: [1, 0, 0, 0],
        1: [0, 1, 0, 1],
        2: [0, 0, 1, 0]
    }
    insert_row_num = simon_algo._add_missing_msb_vector()

    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors
    W_expected = {
        0: [1, 0, 0, 0],
        1: [0, 1, 0, 1],
        2: [0, 0, 1, 0],
        3: [0, 0, 0, 1]
    }
    assert insert_row_num == 3

    np.testing.assert_equal(W_actual, W_expected)


def test_add_row_in_middle():
    simon_algo = Simon()
    simon_algo.n_qubits = 5
    simon_algo._dict_of_linearly_indep_bit_vectors = {
        0: [1, 1, 0, 0, 0],
        2: [0, 0, 1, 0, 1],
        3: [0, 0, 0, 1, 0],
        4: [0, 0, 0, 0, 1]
    }
    insert_row_num = simon_algo._add_missing_msb_vector()

    W_actual = simon_algo._dict_of_linearly_indep_bit_vectors
    W_expected = {
        0: [1, 1, 0, 0, 0],
        1: [0, 1, 0, 0, 0],
        2: [0, 0, 1, 0, 1],
        3: [0, 0, 0, 1, 0],
        4: [0, 0, 0, 0, 1]
    }

    assert insert_row_num == 1

    np.testing.assert_equal(W_actual, W_expected)


def test_bit_map_generation():
    mask = '101'
    expected_map = {
        '000': '101',
        '001': '100',
        '010': '111',
        '011': '110',
        '100': '001',
        '101': '000',
        '110': '011',
        '111': '010'
    }
    actual_map = create_1to1_bitmap(mask)
    assert actual_map == expected_map


def test_2to1_bit_map_generation():
    mask = '101'
    expected_map = {
        '000': '001',
        '101': '001',
        '001': '101',
        '100': '101',
        '010': '000',
        '111': '000',
        '011': '111',
        '110': '111'
    }
    # need to patch numpy as random seed behaves differently on
    # py27 vs. py36
    with patch("numpy.random.choice") as rd_fake:
        rd_fake.return_value = ['001', '101', '000', '111']

        actual_map = create_valid_2to1_bitmap(mask)
        assert actual_map == expected_map


def test_check_mask_correct():
    sa = Simon()

    sa.mask = [1, 1, 0]
    sa.bit_map = {
        '000': '101',
        '001': '010',
        '010': '000',
        '011': '110',

        '100': '000',
        '101': '110',
        '110': '101',
        '111': '010'
    }

    assert sa._check_mask_correct()
