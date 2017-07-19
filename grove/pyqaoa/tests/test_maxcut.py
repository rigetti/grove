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

from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
from grove.pyqaoa.qaoa import QAOA
from grove.pyqaoa.utils import compare_progs
from pyquil.quil import Program
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.gates import H, X, PHASE, CNOT, RZ
import numpy as np
from mock import Mock, patch
import pyquil.api as qvm_mod
import pytest


def test_pass_hamiltonians():
    ref_ham = [PauliSum([PauliTerm("X", 0, -1.0)]), PauliSum([PauliTerm("X", 1,
                                                              -1.0)])]
    cost_ham = [PauliTerm("I", 0, 0.5) + PauliTerm("Z", 0, -0.5) *
                PauliTerm("Z", 1, 1.0)]
    fakeQVM = Mock()
    inst = QAOA(fakeQVM, 2, steps=1,
                cost_ham=cost_ham, ref_hamiltonian=ref_ham)

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
             cost_ham=PauliTerm("X", 0, 1.0), ref_hamiltonian=ref_ham,
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
