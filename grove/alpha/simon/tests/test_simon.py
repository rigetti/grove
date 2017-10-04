"""Test class for helper methods found simon"""

import numpy as np
from grove.alpha.simon.simon import Simon
from pyquil.quil import Program

from mock import patch


expected_return = [
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.]
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
    simon_algo = Simon([0, 2, 2, 0])
    assert simon_algo.n_qubits == 2
    assert simon_algo.n_ancillas == 2
    assert simon_algo.log_qubits == [0, 1]
    assert simon_algo.ancillas == [2, 3]

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.return_value = [[1, 1], [0, 1]]

    mask_string, n_iter, simon_program = simon_algo.find_mask(qvm)
    assert mask_string == '11'
    assert n_iter == 1

    assert simon_program.__str__() == _create_expected_program().__str__()


def test_unitary_function_return():
    simon_algo = Simon([0, 2, 2, 0])
    actual_return = simon_algo._construct_unitary_matrix([0, 2, 2, 0])
    np.testing.assert_equal(actual_return, expected_return)


def test_oracle_program():
    simon_algo = Simon([0, 2, 2, 0])
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
    simon_algo = Simon([0, 2, 2, 0])
    func = simon_algo._construct_unitary_matrix([0, 2, 2, 0])
    orc_func = simon_algo._construct_oracle()

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.return_value = [[1, 1], [0, 1]]
    assert simon_algo.check_two_to_one(qvm, orc_func, [2, 3], "11")


def test_no_substitution():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 0, 1, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0]])
    z = np.array([1, 1, 1, 0, 0])  # linear combination of first two rows

    W = simon_algo.insert_into_row_echelon_binary_matrix(W, z)

    W_expected = np.array([[1, 0, 1, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0]])

    assert np.allclose(W, W_expected)


def test_insert_directly():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 0, 1, 0]])
    z = np.array([0, 0, 1, 0, 1])

    W = simon_algo.insert_into_row_echelon_binary_matrix(W, z)
    W_expected = np.array([[1, 1, 0, 0, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 1]])

    assert np.allclose(W, W_expected)


def test_insert_after_xor():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0]])

    z = np.array([1, 0, 1, 0, 1, 1])

    W = simon_algo.insert_into_row_echelon_binary_matrix(W, z)
    W_expected = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1, 1]])

    assert np.allclose(W, W_expected)


def test_add_row_at_top():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]])
    W, insert_row_num = simon_algo.make_square_row_echelon(W)

    W_expected = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 1]])

    assert insert_row_num == 0

    assert np.allclose(W, W_expected)


def test_add_row_at_bottom():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0]])
    W, insert_row_num = simon_algo.make_square_row_echelon(W)

    W_expected = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    assert insert_row_num == 3

    assert np.allclose(W, W_expected)


def test_add_row_in_middle():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])
    W, insert_row_num = simon_algo.make_square_row_echelon(W)

    W_expected = np.array([[1, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 1],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])

    assert insert_row_num == 1

    assert np.allclose(W, W_expected)


def test_one_at_top():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    s = np.array([1, 0, 0, 0, 0])
    x = simon_algo.binary_back_substitute(W, s)

    prod = np.dot(W, x)
    prod = np.vectorize(lambda x: x % 2)(prod)

    assert np.allclose(s, prod)


def test_one_at_bottom():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    s = np.array([0, 0, 0, 1])
    x = simon_algo.binary_back_substitute(W, s)

    prod = np.dot(W, x)
    prod = np.vectorize(lambda x: x % 2)(prod)

    assert np.allclose(s, prod)


def test_one_at_middle():
    simon_algo = Simon([0, 2, 2, 0])
    W = np.array([[1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]])

    s = np.array([0, 1, 0, 0, 0])
    x = simon_algo.binary_back_substitute(W, s)

    prod = np.dot(W, x)
    prod = np.vectorize(lambda x: x % 2)(prod)

    assert np.allclose(s, prod)
