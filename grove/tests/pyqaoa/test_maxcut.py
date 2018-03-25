##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import pytest
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
from grove.pyqaoa.qaoa import QAOA
from pyquil.quil import Program
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.gates import H, X, PHASE, CNOT, RZ
import numpy as np
from mock import Mock, patch
import pyquil.api as qvm_mod


def test_pass_hamiltonians():
    ref_ham = [PauliSum([PauliTerm("X", 0, -1.0)]), PauliSum([PauliTerm("X", 1,
                                                              -1.0)])]
    cost_ham = [PauliTerm("I", 0, 0.5) + PauliTerm("Z", 0, -0.5) *
                PauliTerm("Z", 1, 1.0)]
    fakeQVM = Mock()
    inst = QAOA(fakeQVM, range(2), steps=1,
                cost_ham=cost_ham, ref_ham=ref_ham)

    c = inst.cost_ham
    r = inst.ref_ham
    assert isinstance(c, list)
    assert isinstance(r, list)
    assert isinstance(c[0], PauliSum)
    assert isinstance(r[0], PauliSum)
    assert len(c) == 1
    assert len(r) == 2

    with pytest.raises(TypeError):
        QAOA(fakeQVM, 2, steps=1,
             cost_ham=PauliTerm("X", 0, 1.0), ref_ham=ref_ham,
             rand_seed=42)


def test_hamiltonians():
    test_graph = [(0, 1)]
    p = 1
    inst = maxcut_qaoa(test_graph, steps=p)

    cost_ops, ref_func = inst.cost_ham, inst.ref_ham
    for op in cost_ops:
        for term in op.terms:
            assert(np.isclose(np.abs(term.coefficient), 0.5))

    assert len(ref_func) == 2
    assert len(cost_ops) == 1


def test_param_prog_p1_barbell():
    test_graph = [(0, 1)]
    p = 1
    with patch('grove.pyqaoa.maxcut_qaoa.api', spec=qvm_mod):
        inst = maxcut_qaoa(test_graph, steps=p)

        param_prog = inst.get_parameterized_program()
        trial_prog = param_prog([1.2, 3.4])
        result_prog = Program().inst([H(0), H(1), X(0), PHASE(1.7)(0), X(0),
                                     PHASE(1.7)(0), CNOT(0, 1), RZ(3.4)(1),
                                     CNOT(0, 1), H(0), RZ(-2.4)(0), H(0), H(1),
                                     RZ(-2.4)(1), H(1)])
        trial_prog == result_prog


def test_psiref_bar_p2():
    bar = [(0, 1)]
    p = 2
    with patch('grove.pyqaoa.maxcut_qaoa.api', spec=qvm_mod):
        inst = maxcut_qaoa(bar, steps=p)

    param_prog = inst.get_parameterized_program()

    # returns are the rotations correct?
    prog = param_prog([1.2, 3.4, 2.1, 4.5])
    result_prog = Program().inst([H(0), H(1),
                                  X(0), PHASE(1.05)(0), X(0), PHASE(1.05)(0),
                                  CNOT(0, 1), RZ(2.1)(1), CNOT(0, 1),
                                  H(0), RZ(-2.4)(0), H(0),
                                  H(1), RZ(-2.4)(1), H(1),
                                  X(0), PHASE(2.25)(0), X(0), PHASE(2.25)(0),
                                  CNOT(0, 1), RZ(4.5)(1), CNOT(0, 1),
                                  H(0), RZ(-6.8)(0), H(0),
                                  H(1), RZ(-6.8)(1), H(1),
                                  ])
    assert prog == result_prog


if __name__ == "__main__":
    test_psiref_bar_p2()
