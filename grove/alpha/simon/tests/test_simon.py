"""Test class for helper methods found simon"""

import numpy as np
from grove.alpha.simon.simon import Simon, create_periodic_bitmap
import grove.alpha.simon.utils as u
from pyquil.quil import Program

from mock import patch


expected_return = [
    [0., 0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 1.],
    [0., 0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 0.]
]


def _create_expected_program():
    expected_prog = Program()
    expected_prog.defgate("FUNCT", expected_return)
    expected_prog.defgate("FUNCT-INV", np.linalg.inv(expected_return))
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")

    expected_prog.inst("FUNCT 4 0 1")
    expected_prog.inst("CNOT 0 2")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("FUNCT-INV 4 0 1")

    expected_prog.inst("H 0")
    expected_prog.inst("H 1")

    return expected_prog


def test_simon_class():
    simon_algo = Simon()

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.return_value = [[0, 1]]

    bit_string_mapping = {
        '00': '10',
        '01': '11',
        '10': '00',
        '11': '01'
    }

    n_iter, mask, is_2to1 = simon_algo.find_mask(qvm, bit_string_mapping)

    assert simon_algo.n_qubits == 2
    assert simon_algo.n_ancillas == 2
    assert simon_algo.log_qubits == [0, 1]
    assert simon_algo.ancillas == [2, 3]

    assert mask == [1, 0]
    assert n_iter == 1
    assert is_2to1

    assert simon_algo.simon_circuit.__str__() == _create_expected_program().__str__()


def test_unitary_function_return():
    simon_algo = Simon()
    bit_string_mapping = {
        '00': '10',
        '01': '11',
        '10': '00',
        '11': '01'
    }

    actual_return = simon_algo._construct_unitary_matrix(u.mapping_dict_to_list(bit_string_mapping))
    np.testing.assert_equal(actual_return, expected_return)


def test_oracle_program():
    simon_algo = Simon()

    simon_algo.n_qubits = 2
    simon_algo.n_ancillas = 2
    simon_algo.log_qubits = [0, 1]
    simon_algo.ancillas = [2, 3]
    simon_algo.unitary_function_mapping = expected_return

    actual_prog = simon_algo._construct_oracle()
    expected_prog = Program()
    expected_prog.defgate("FUNCT", expected_return)
    expected_prog.defgate("FUNCT-INV", np.linalg.inv(expected_return))
    expected_prog.inst("FUNCT 4 0 1")
    expected_prog.inst("CNOT 0 2")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("FUNCT-INV 4 0 1")
    assert expected_prog.__str__() == actual_prog.__str__()


def test_check_two_to_one():
    simon_algo = Simon()

    simon_algo.n_qubits = 2
    simon_algo.n_ancillas = 2
    simon_algo.log_qubits = [0, 1]
    simon_algo.ancillas = [2, 3]
    simon_algo.unitary_function_mapping = expected_return
    simon_algo.oracle_circuit = simon_algo._construct_oracle()
    simon_algo.mask = [1, 1]

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.return_value = [[1, 1], [0, 1]]
    assert simon_algo.check_two_to_one(qvm)


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
    insert_row_num = simon_algo._add_missing_provenance_vector()

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
    insert_row_num = simon_algo._add_missing_provenance_vector()

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
    insert_row_num = simon_algo._add_missing_provenance_vector()

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
    actual_map = create_periodic_bitmap(mask)
    assert actual_map == expected_map

