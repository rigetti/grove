import numpy as np
import pyquil.api as api
import pyquil.quil as pq
import pytest
from pyquil.gates import X
from mock import patch

from pyquil.api import SyncConnection
from pyquil.quil import Program

from grove.alpha.bernstein_vazirani.bernstein_vazirani import BernsteinVazirani


def _create_expected_prog():
    expected_prog = Program()
    expected_prog.inst("X 3")
    expected_prog.inst("H 3")
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    expected_prog.inst("H 2")
    expected_prog.inst("X 3")
    expected_prog.inst("CNOT 2 3")
    expected_prog.inst("CNOT 1 3")
    expected_prog.inst("H 0")
    expected_prog.inst("H 1")
    expected_prog.inst("H 2")
    return expected_prog


# def test_bernstein_vazirani():
#     orc = oracle_function([1, 1, 0], 1, [0, 1, 2], 3)
#     bv_prog = bernstein_vazirani(orc, [0, 1, 2], 3)
#
#     with patch("pyquil.api.SyncConnection") as qvm:
#         # Need to mock multiple returns as an iterable
#         qvm.run_and_measure.side_effect = [
#             ([0, 1, 1], ),
#             ([1], )
#         ]
#
#
#     # qvm = SyncConnection("http://127.0.0.1:5000")
#
#     bv_a, bv_b, bv_program = run_bernstein_vazirani(qvm, orc, [0, 1, 2], 3)
#
#     assert bv_a == [1, 1, 0]
#     assert bv_b == 1
#     assert bv_program.__str__() == _create_expected_prog().__str__()


def test_bv_class():

    with patch("pyquil.api.SyncConnection") as qvm:
        # Need to mock multiple returns as an iterable
        qvm.run_and_measure.side_effect = [
            ([0, 1, 1], ),
            ([1], )
        ]

    # qvm = SyncConnection("http://127.0.0.1:5000")
    bv = BernsteinVazirani()
    bv_a, bv_b = bv.with_oracle_for_vector([1, 1, 0], 1).run(qvm)

    assert bv_a == [1, 1, 0]
    assert bv_b == 1
    assert bv.full_bv_circuit.__str__() == _create_expected_prog().__str__()
