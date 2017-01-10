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

import numpy as np
from grove.pyqaoa.qaoa import QAOA
import pyquil.forest as qvm_module
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.gates import X, Y, Z
from pyquil.quil import Program
from grove.pyvqe.vqe import VQE
from mock import Mock, patch


def isclose(a, b, rel_tol=1e-10, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compare_progs(test, reference):
    """
    compares two programs gate by gate, param by param
    """
    tinstr = test.instructions
    rinstr = reference.instructions
    assert len(tinstr) == len(rinstr)
    for idx in xrange(len(tinstr)):
        # check each field of the instruction object
        assert tinstr[idx].operator_name == rinstr[idx].operator_name
        assert len(tinstr[idx].parameters) == len(rinstr[idx].parameters)
        for pp in xrange(len(tinstr[idx].parameters)):
            cmp_val = isclose(tinstr[idx].parameters[pp], rinstr[idx].parameters[pp])
            assert cmp_val

        assert len(tinstr[idx].arguments) == len(rinstr[idx].arguments)
        for aa in xrange(len(tinstr[idx].arguments)):
            assert tinstr[idx].arguments[aa] == rinstr[idx].arguments[aa]


def test_probabilities():
    p = 1
    n_qubits = 2
    # known set of angles for barbell
    angles = [1.96348709,  4.71241069]
    wf = np.array([-1.17642098e-05 - 1j*7.67538040e-06,
                   -7.67563580e-06 - 1j*7.07106781e-01,
                   -7.67563580e-06 - 1j*7.07106781e-01,
                   -1.17642098e-05 - 1j*7.67538040e-06])
    fakeQVM = Mock(spec=qvm_module.Connection())
    fakeQVM.wavefunction = Mock(return_value=wf)
    inst = QAOA(fakeQVM, n_qubits, steps=p,
                rand_seed=42)

    true_probs = np.zeros_like(wf)
    for xx in xrange(wf.shape[0]):
        true_probs[xx] = np.conj(wf[xx]) * wf[xx]
    probs = inst.probabilities(angles)
    assert isinstance(probs, np.ndarray)
    prob_true = np.zeros((2**inst.n_qubits, 1))
    prob_true[1] = 0.5
    prob_true[2] = 0.5
    assert np.isclose(probs, prob_true).all()


def test_get_angles():
    p = 2
    n_qubits = 2
    fakeQVM = Mock()
    with patch('grove.pyqaoa.qaoa.VQE', spec=VQE) as mockVQEClass:
        inst = mockVQEClass.return_value
        result = Mock()
        result.x = [1.2, 2.1, 3.4, 4.3]
        inst.vqe_run.return_value = result
        MCinst = QAOA(fakeQVM, n_qubits, steps=p,
                      cost_ham=[PauliSum([PauliTerm("X", 0)])])
        betas, gammas = MCinst.get_angles()
        assert betas == [1.2, 2.1]
        assert gammas == [3.4, 4.3]


def test_ref_program_pass():
    ref_prog = Program().inst([X(0), Y(1), Z(2)])
    fakeQVM = Mock(spec=qvm_module.Connection())
    inst = QAOA(fakeQVM, 2, driver_ref=ref_prog)
    param_prog = inst.get_parameterized_program()
    test_prog = param_prog([0, 0])
    compare_progs(ref_prog, test_prog)
